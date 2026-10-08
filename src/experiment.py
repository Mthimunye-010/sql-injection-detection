"""Experiment orchestration shared by `scripts/run_experiments.py` and the
notebook, so that both produce byte-identical numbers from one code path.

Protocol
--------
* The decision threshold of every *scored* method is selected on `val` and then
  applied unchanged to `test_id` and `test_ood`.  The static analysers are
  rule-based and emit a hard decision, so no threshold applies to them.
* Neural variants are trained with 3 seeds; tables report mean +/- sd and
  paired significance tests across the matched seeds.
* Two retrieval tasks are evaluated (see `retrieval_tasks`).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .baselines import (build_tfidf_lr, static_regex_predict,
                        static_taint_predict)
from .data import EncodedSplit, subsample, token_strings
from .metrics import (attention_metrics, best_threshold, detection_metrics,
                      paired_tests, retrieval_metrics)
from .models import VARIANT_LABELS, VARIANTS
from .train import TrainConfig, predict, train_variant

TEST_SPLITS = ("test_id", "test_ood")

METHOD_LABELS = {
    "M1_static_regex": "Static analyser -- pattern rules (semgrep/Bandit class)",
    "M2_static_taint": "Static analyser -- AST taint analysis",
    "M3_tfidf_lr": "TF-IDF + logistic regression",
    **VARIANT_LABELS,
}
METHOD_ORDER = ["M1_static_regex", "M2_static_taint", "M3_tfidf_lr",
                "M4_CNN_meanpool", "M5_CNN_attention",
                "M6_CNN_attn_contrastive", "M7_AGNCS_full"]


# --------------------------------------------------------------------------- #
# Retrieval protocol
# --------------------------------------------------------------------------- #

def retrieval_sets(split: EncodedSplit, n_corpus: int = 6000,
                   n_query: int = 1500, seed: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified corpus indices and the positions within it used as queries."""
    rng = np.random.default_rng(seed)
    corpus = subsample(split, n_corpus, seed=seed).index
    corpus = np.asarray(sorted(set(corpus.tolist())), dtype=int)
    qpos = rng.permutation(len(corpus))[:min(n_query, len(corpus))]
    return corpus, np.sort(qpos)


def eval_retrieval(emb: np.ndarray, split: EncodedSplit, seed: int = 0
                   ) -> Dict[str, float]:
    """Task A -- query-by-example search with *same vulnerability family* as the
    relevance criterion.  Leave-one-out: the query is removed from its own
    result list."""
    corpus, qpos = retrieval_sets(split, seed=seed)
    ce = emb[corpus]
    cf = split.fam[corpus]
    return retrieval_metrics(ce[qpos], ce, cf[qpos], cf, ks=(1, 5, 10),
                             exclude_self=qpos)


def eval_cross_pattern_retrieval(emb_q: np.ndarray, split_q: EncodedSplit,
                                 emb_c: np.ndarray, split_c: EncodedSplit,
                                 n_query: int = 1000, n_corpus: int = 6000,
                                 seed: int = 0) -> Dict[str, float]:
    """Task B -- transfer search.

    Query with a vulnerable snippet from a *held-out* family and search a corpus
    drawn from the families seen in training.  No same-family match exists, so
    relevance is "is also vulnerable".  This is the operational use case: an
    analyst finds one new injection pattern and asks the index for everything
    else in the codebase that is dangerous for the same reason.
    """
    rng = np.random.default_rng(seed)
    qidx = np.flatnonzero(split_q.y == 1)
    qidx = rng.permutation(qidx)[:n_query]
    cidx = subsample(split_c, n_corpus, seed=seed).index
    cidx = np.asarray(sorted(set(cidx.tolist())), dtype=int)

    sim = emb_q[qidx] @ emb_c[cidx].T
    rel = (split_c.y[cidx] == 1)[None, :].repeat(len(qidx), axis=0)
    out: Dict[str, float] = {}
    kmax = 10
    top = np.empty((len(sim), kmax), dtype=np.int64)
    for start in range(0, len(sim), 64):
        stop = min(start + 64, len(sim))
        top[start:stop] = np.argsort(-sim[start:stop], axis=1)[:, :kmax]
    row = np.arange(len(qidx))[:, None]
    rel_top = rel[row, top]
    for k in (1, 5, 10):
        out["vuln_p@%d" % k] = float(rel_top[:, :k].mean())
    first = np.argmax(rel_top, axis=1).astype(float)
    hit = rel_top.any(axis=1)
    out["vuln_mrr"] = float(np.where(hit, 1.0 / (first + 1.0), 0.0).mean())
    return out


# --------------------------------------------------------------------------- #
# Static analysers
# --------------------------------------------------------------------------- #

def run_static_baselines(frames: Dict[str, pd.DataFrame],
                         verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    rows, preds = [], {}
    for name, fn in (("M1_static_regex", static_regex_predict),
                     ("M2_static_taint", static_taint_predict)):
        for sp in TEST_SPLITS:
            f = frames[sp]
            t0 = time.time()
            p = np.array([fn(c) for c in f["code"].values], dtype=float)
            dt = time.time() - t0
            preds[(name, sp)] = p
            m = detection_metrics(f["label"].values, p, thr=0.5)
            # A rule-based analyser emits a hard decision, so threshold-free
            # ranking metrics are not defined for it.
            m["roc_auc"], m["pr_auc"] = float("nan"), float("nan")
            rows.append({"method": name, "seed": 0, "split": sp,
                         "ms_per_sample": round(dt / len(f) * 1000, 3), **m})
            if verbose:
                print("  %-16s %-9s F1=%.4f FPR=%.4f rec=%.4f"
                      % (name, sp, m["f1"], m["fpr"], m["recall"]))
    return pd.DataFrame(rows), preds


# --------------------------------------------------------------------------- #
# TF-IDF baseline
# --------------------------------------------------------------------------- #

def run_tfidf_baseline(frames: Dict[str, pd.DataFrame],
                       splits: Dict[str, EncodedSplit],
                       seeds: Sequence[int] = (0, 1, 2),
                       n_train: Optional[int] = None,
                       verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """TF-IDF + logistic regression over the shared token stream.

    Also doubles as a *code-search* baseline: the L2-normalised TF-IDF vectors
    are a classical retrieval index, which is the fair non-neural comparison
    for the search metrics.
    """
    from sklearn.preprocessing import normalize

    if verbose:
        print("  tokenising for TF-IDF ...")
    texts = {sp: token_strings(frames[sp]) for sp in
             ("train", "val", "test_id", "test_ood")}
    y_train = frames["train"]["label"].values

    rows, artefacts = [], {}
    for seed in seeds:
        idx = np.arange(len(y_train))
        if n_train is not None and n_train < len(idx):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(idx)[:n_train]
        t0 = time.time()
        pipe = build_tfidf_lr([texts["train"][i] for i in idx], y_train[idx],
                              seed=seed)
        fit_s = time.time() - t0

        pv = pipe.predict_proba(texts["val"])[:, 1]
        thr, _ = best_threshold(frames["val"]["label"].values, pv, "f1")

        for sp in TEST_SPLITS:
            p = pipe.predict_proba(texts[sp])[:, 1]
            m = detection_metrics(frames[sp]["label"].values, p, thr=thr)
            rows.append({"method": "M3_tfidf_lr", "seed": seed, "split": sp,
                         "threshold": thr, "fit_seconds": round(fit_s, 1), **m})
            artefacts[("prob", "M3_tfidf_lr", seed, sp)] = p
            if seed == seeds[0]:
                vec = pipe.named_steps["tfidf"].transform(texts[sp])
                artefacts[("embed", "M3_tfidf_lr", seed, sp)] = normalize(vec)
        if verbose:
            last = rows[-1]
            print("  M3_tfidf_lr seed=%d  test_ood F1=%.4f FPR=%.4f"
                  % (seed, last["f1"], last["fpr"]))
    return pd.DataFrame(rows), artefacts


def tfidf_retrieval(artefacts: dict, splits: Dict[str, EncodedSplit],
                    seed: int = 0) -> List[dict]:
    """Sparse cosine retrieval with the TF-IDF index."""
    rows = []
    for sp in TEST_SPLITS:
        key = ("embed", "M3_tfidf_lr", seed, sp)
        if key not in artefacts:
            continue
        X = artefacts[key]
        s = splits[sp]
        corpus, qpos = retrieval_sets(s, seed=0)
        C = X[corpus]
        sim = (C[qpos] @ C.T).toarray()
        cf = s.fam[corpus]
        rm = _retrieval_from_sim(sim, cf[qpos], cf, qpos)
        rows.append({"method": "M3_tfidf_lr", "seed": seed, "split": sp, **rm})
    return rows


def _retrieval_from_sim(sim: np.ndarray, qfam: np.ndarray, cfam: np.ndarray,
                        self_pos: np.ndarray) -> Dict[str, float]:
    sim = sim.copy()
    sim[np.arange(len(sim)), self_pos] = -np.inf
    rel = (qfam[:, None] == cfam[None, :])
    rel[np.arange(len(sim)), self_pos] = False
    kmax = 10
    top = np.argsort(-sim, axis=1)[:, :kmax]
    row = np.arange(len(sim))[:, None]
    rel_top = rel[row, top]
    out = {}
    for k in (1, 5, 10):
        out["recall@%d" % k] = float(rel_top[:, :k].any(axis=1).mean())
        out["p@%d" % k] = float(rel_top[:, :k].mean())
    first = np.argmax(rel_top, axis=1).astype(float)
    hit = rel_top.any(axis=1)
    out["mrr"] = float(np.where(hit, 1.0 / (first + 1.0), 0.0).mean())
    disc = 1.0 / np.log2(np.arange(1, kmax + 1) + 1.0)
    dcg = (rel_top * disc).sum(axis=1)
    n_rel = rel.sum(axis=1)
    idcg = np.array([disc[:min(int(r), kmax)].sum() for r in n_rel])
    out["ndcg@10"] = float(np.where(idcg > 0, dcg / idcg, 0.0).mean())
    # Compute full-ranking MAP in row chunks.  Holding argsort indices,
    # cumulative counts and precision for every query at once can require
    # several extra copies of the query-by-corpus matrix.
    map_sum = 0.0
    map_chunk = 64
    rank = np.arange(1, sim.shape[1] + 1, dtype=np.float32)[None, :]
    for start in range(0, len(sim), map_chunk):
        stop = min(start + map_chunk, len(sim))
        order_chunk = np.argsort(-sim[start:stop], axis=1)
        row_chunk = np.arange(stop - start)[:, None]
        rel_sorted = rel[start:stop][row_chunk, order_chunk]
        csum = np.cumsum(rel_sorted, axis=1, dtype=np.int32)
        prec_at = csum / rank
        ap = np.where(n_rel[start:stop] > 0,
                      (prec_at * rel_sorted).sum(axis=1) /
                      np.maximum(n_rel[start:stop], 1), 0.0)
        map_sum += float(ap.sum())
    out["map"] = map_sum / max(len(sim), 1)
    return out


# --------------------------------------------------------------------------- #
# Neural variants
# --------------------------------------------------------------------------- #

def run_neural(splits: Dict[str, EncodedSplit], vocab_size: int, max_len: int,
               variants: Sequence[str], seeds: Sequence[int],
               cfg_kwargs: dict, n_train: Optional[int] = None,
               n_val: Optional[int] = None, keep_alpha_seed: int = 0,
               verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    det_rows: List[dict] = []
    artefacts: dict = {}

    val = splits["val"] if n_val is None else subsample(splits["val"], n_val, seed=0)

    for variant in variants:
        for seed in seeds:
            tr = (splits["train"] if n_train is None
                  else subsample(splits["train"], n_train, seed=seed))
            cfg = TrainConfig(seed=seed, verbose=verbose, **cfg_kwargs)
            if verbose:
                print("\n  [%s] seed=%d  n_train=%d" % (variant, seed, len(tr)))
            model, hist, info = train_variant(variant, tr, val, vocab_size,
                                              max_len, cfg)
            artefacts[("history", variant, seed)] = hist
            artefacts[("info", variant, seed)] = info

            pv = predict(model, val, want_alpha=False)
            thr, _ = best_threshold(val.y, pv["prob"], "f1")

            want_a = (model.pooling_kind == "attention")
            for sp in TEST_SPLITS:
                s = splits[sp]
                p = predict(model, s, want_alpha=want_a)
                dm = detection_metrics(s.y, p["prob"], thr=thr)
                row = {"method": variant, "seed": seed, "split": sp,
                       "threshold": thr, "n_params": info["n_params"],
                       "best_epoch": info["best_epoch"],
                       "train_seconds": round(info["train_seconds"], 1), **dm}
                if want_a:
                    row.update(attention_metrics(p["alpha"], s.mask, s.valid))
                row.update(eval_retrieval(p["embed"], s, seed=0))
                det_rows.append(row)
                artefacts[("prob", variant, seed, sp)] = p["prob"]
                artefacts[("embed", variant, seed, sp)] = p["embed"]
                if seed == keep_alpha_seed and want_a:
                    artefacts[("alpha", variant, seed, sp)] = p["alpha"]

            # Task B: transfer search, OOD queries against the ID corpus.
            xb = eval_cross_pattern_retrieval(
                artefacts[("embed", variant, seed, "test_ood")], splits["test_ood"],
                artefacts[("embed", variant, seed, "test_id")], splits["test_id"])
            for r in det_rows[-len(TEST_SPLITS):]:
                if r["split"] == "test_ood":
                    r.update(xb)
            if seed == keep_alpha_seed:
                artefacts[("model", variant, seed)] = model

    return pd.DataFrame(det_rows), artefacts


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #

MAIN_METRICS = ["f1", "fpr", "recall", "precision", "accuracy", "mcc",
                "roc_auc", "pr_auc", "balanced_accuracy"]
SEARCH_METRICS = ["mrr", "recall@1", "recall@10", "map", "ndcg@10"]
ATT_METRICS = ["att_mass_rationale", "att_lift", "att_pointing", "att_iou"]


def summarise(df: pd.DataFrame, metrics: Sequence[str] = MAIN_METRICS
              ) -> pd.DataFrame:
    """mean +/- sd over seeds, one row per (method, split)."""
    have = [m for m in metrics if m in df.columns]
    g = df.groupby(["method", "split"])[have].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=g.index)
    for m in have:
        mu, sd = g[(m, "mean")], g[(m, "std")].fillna(0.0)
        out[m] = mu.round(4)
        out[m + "_sd"] = sd.round(4)
    out = out.reset_index()
    out["order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values(["split", "order"]).drop(columns="order")


def pretty_table(summary: pd.DataFrame, split: str,
                 metrics: Sequence[str] = ("f1", "fpr", "recall", "precision",
                                           "roc_auc", "mcc")) -> pd.DataFrame:
    sub = summary[summary["split"] == split].copy()
    rows = []
    for _, r in sub.iterrows():
        row = {"Method": METHOD_LABELS.get(r["method"], r["method"])}
        for m in metrics:
            if m not in r or pd.isna(r[m]):
                row[m.upper()] = "--"
            elif r.get(m + "_sd", 0) and r[m + "_sd"] > 0:
                row[m.upper()] = "%.4f ± %.4f" % (r[m], r[m + "_sd"])
            else:
                row[m.upper()] = "%.4f" % r[m]
        rows.append(row)
    return pd.DataFrame(rows)


def significance_table(df: pd.DataFrame, baseline: str, proposed: str,
                       metrics: Sequence[str] = ("f1", "fpr", "mrr", "att_lift")
                       ) -> pd.DataFrame:
    rows = []
    for split in TEST_SPLITS:
        a = df[(df.method == proposed) & (df.split == split)].sort_values("seed")
        b = df[(df.method == baseline) & (df.split == split)].sort_values("seed")
        for m in metrics:
            if m not in a.columns or a[m].isna().all() or b[m].isna().all():
                continue
            t = paired_tests(a[m].values, b[m].values)
            rows.append({"split": split, "metric": m,
                         "proposed": round(t["mean_a"], 4),
                         "baseline": round(t["mean_b"], 4),
                         "diff": round(t["mean_diff"], 4),
                         "t_p": round(t["t_p"], 5) if t["t_p"] == t["t_p"] else float("nan"),
                         "wilcoxon_p": round(t["wilcoxon_p"], 5) if t["wilcoxon_p"] == t["wilcoxon_p"] else float("nan"),
                         "cohen_dz": round(t["cohen_dz"], 3),
                         "n_seeds": t["n"]})
    return pd.DataFrame(rows)


def per_family_table(frames: Dict[str, pd.DataFrame], preds: dict,
                     artefacts: dict, methods: Sequence[str],
                     thresholds: Dict[str, float], seed: int = 0
                     ) -> pd.DataFrame:
    """Error rate of every method on every family -- the diagnostic view."""
    rows = []
    for sp in TEST_SPLITS:
        if sp not in frames:
            continue
        f = frames[sp]
        for meth in methods:
            if meth.startswith("M1") or meth.startswith("M2"):
                p = preds.get((meth, sp))
                if p is None:
                    continue
                pred = (p >= 0.5).astype(int)
            else:
                pr = artefacts.get(("prob", meth, seed, sp))
                if pr is None:
                    continue
                pred = (pr >= thresholds.get((meth, seed), 0.5)).astype(int)
            tmp = f.assign(pred=pred)
            g = tmp.groupby("family").apply(
                lambda d: pd.Series({
                    "label": int(d["label"].iloc[0]),
                    "n": len(d),
                    "fire_rate": float(d["pred"].mean()),
                    "error_rate": float((d["pred"] != d["label"]).mean()),
                }), include_groups=False)
            for fam, r in g.iterrows():
                rows.append({"split": sp, "method": meth, "family": fam,
                             "label": int(r["label"]), "n": int(r["n"]),
                             "fire_rate": round(r["fire_rate"], 4),
                             "error_rate": round(r["error_rate"], 4)})
    return pd.DataFrame(rows)


def per_family_table_multiseed(frames: Dict[str, pd.DataFrame], preds: dict,
                               artefacts: dict, methods: Sequence[str],
                               thresholds: Dict[str, float],
                               seeds: Sequence[int]) -> pd.DataFrame:
    """Per-family error rates averaged over seeds, with their spread.

    A single seed's per-family breakdown is not safe to read on the held-out
    split: the seed-to-seed spread there is large (see the receptive-field
    ablation), so a family can look solved or broken by luck.  Averaging and
    reporting the standard deviation keeps the diagnostic honest.  The static
    analysers are deterministic, so their rows simply carry sd = 0.
    """
    frames_list = [per_family_table(frames, preds, artefacts, methods,
                                    thresholds, seed=s) for s in seeds]
    cat = pd.concat(frames_list, ignore_index=True)
    out = (cat.groupby(["split", "method", "family", "label"], as_index=False)
              .agg(n=("n", "first"),
                   fire_rate=("fire_rate", "mean"),
                   error_rate=("error_rate", "mean"),
                   error_sd=("error_rate", "std"),
                   n_seeds=("error_rate", "size")))
    out["error_sd"] = out["error_sd"].fillna(0.0)
    for c in ("fire_rate", "error_rate", "error_sd"):
        out[c] = out[c].round(4)
    return out


def hard_negative_fpr(frames: Dict[str, pd.DataFrame], preds: dict,
                      artefacts: dict, methods: Sequence[str],
                      thresholds: Dict[str, float], seeds: Sequence[int]
                      ) -> pd.DataFrame:
    """False-positive rate restricted to the hard-negative families.

    This is the number that decides whether a detector is deployable: plain
    safe code is easy, and an analyst's time is burned by confident alerts on
    code that merely *looks* dangerous.
    """
    rows = []
    for sp in TEST_SPLITS:
        if sp not in frames:
            continue
        f = frames[sp]
        hn = (f["hard_negative"].values == 1) & (f["label"].values == 0)
        easy = (f["hard_negative"].values == 0) & (f["label"].values == 0)
        if hn.sum() == 0:
            continue
        for meth in methods:
            use_seeds = [0] if meth.startswith(("M1", "M2")) else list(seeds)
            for seed in use_seeds:
                if meth.startswith(("M1", "M2")):
                    p = preds.get((meth, sp))
                    if p is None:
                        continue
                    pred = (p >= 0.5).astype(int)
                else:
                    pr = artefacts.get(("prob", meth, seed, sp))
                    if pr is None:
                        continue
                    pred = (pr >= thresholds.get((meth, seed), 0.5)).astype(int)
                rows.append({
                    "split": sp, "method": meth, "seed": seed,
                    "fpr_hard_negatives": round(float(pred[hn].mean()), 4),
                    "fpr_easy_negatives": round(float(pred[easy].mean()), 4)
                    if easy.sum() else float("nan"),
                    "n_hard": int(hn.sum()), "n_easy": int(easy.sum()),
                })
    return pd.DataFrame(rows)
