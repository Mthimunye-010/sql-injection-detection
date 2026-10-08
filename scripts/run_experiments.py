"""Run the full comparative study and write every table and figure.

    python scripts/run_experiments.py                 # full study (3 seeds)
    python scripts/run_experiments.py --quick         # 1 seed, small subsets

Everything lands in `results/` and is consumed unchanged by the notebook and
the report.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import viz                                                   # noqa: E402
from src.data import prepare, subsample                               # noqa: E402
from src.experiment import (ATT_METRICS, MAIN_METRICS, METHOD_ORDER,  # noqa: E402
                            SEARCH_METRICS, hard_negative_fpr,
                            per_family_table_multiseed, pretty_table,
                            run_neural, run_static_baselines,
                            run_tfidf_baseline, significance_table,
                            summarise, tfidf_retrieval)
from src.models import VARIANTS                                        # noqa: E402

TABLES = "results/tables"
TEST_SPLITS_FIG = ("test_id", "test_ood")
FIGS = "results/figures"
RAW = "results/raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4],
                    help="dilation schedule; receptive field = 1 + 2*sum")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.n_train, args.epochs = [0], 4000, 3

    torch.set_num_threads(args.threads)
    for d in (TABLES, FIGS, RAW):
        os.makedirs(d, exist_ok=True)
    viz.set_style()

    t_all = time.time()
    print("=" * 72)
    print("NCOS734 PG4 -- Attention-Guided Neural Code Search for SQLi detection")
    print("=" * 72)

    df = pd.read_parquet("data/sqli_corpus.parquet")
    splits, vocab, frames = prepare(df, max_len=args.max_len, cache_dir="data/cache")
    print("\nvocabulary: %d tokens | max_len: %d | d_model: %d"
          % (len(vocab), args.max_len, args.d_model))

    # ------------------------------------------------------------- baselines
    print("\n[1/5] static analysers")
    static_df, static_preds = run_static_baselines(frames)

    print("\n[2/5] TF-IDF + logistic regression")
    tfidf_df, tfidf_art = run_tfidf_baseline(frames, splits, seeds=args.seeds,
                                             n_train=args.n_train)
    tfidf_ret = pd.DataFrame(tfidf_retrieval(tfidf_art, splits, seed=args.seeds[0]))
    if len(tfidf_ret):
        tfidf_df = tfidf_df.merge(tfidf_ret, on=["method", "seed", "split"],
                                  how="left")

    # ---------------------------------------------------------------- neural
    print("\n[3/5] neural variants (%d variants x %d seeds)"
          % (len(VARIANTS), len(args.seeds)))
    cfg_kwargs = dict(epochs=args.epochs, batch_size=128, d_model=args.d_model,
                      patience=2, lr=2.5e-3, families_per_batch=16,
                      dilations=tuple(args.dilations))
    neural_df, art = run_neural(splits, len(vocab), args.max_len,
                                variants=list(VARIANTS), seeds=args.seeds,
                                cfg_kwargs=cfg_kwargs, n_train=args.n_train,
                                n_val=4000, keep_alpha_seed=args.seeds[0])

    all_df = pd.concat([static_df, tfidf_df, neural_df], ignore_index=True)
    all_df.to_csv(os.path.join(TABLES, "all_runs.csv"), index=False)

    # -------------------------------------------------------------- summarise
    print("\n[4/5] tables")
    summary = summarise(all_df, MAIN_METRICS + SEARCH_METRICS + ATT_METRICS
                        + ["vuln_p@10", "vuln_mrr"])
    summary.to_csv(os.path.join(TABLES, "summary_by_method_split.csv"), index=False)

    for sp in ("test_id", "test_ood"):
        t = pretty_table(summary, sp)
        t.to_csv(os.path.join(TABLES, "table_detection_%s.csv" % sp), index=False)
        print("\n--- Detection, %s ---" % sp)
        print(t.to_string(index=False))
        ts = pretty_table(summary, sp, metrics=SEARCH_METRICS)
        ts.to_csv(os.path.join(TABLES, "table_search_%s.csv" % sp), index=False)
        ta = pretty_table(summary, sp, metrics=ATT_METRICS)
        ta.to_csv(os.path.join(TABLES, "table_attention_%s.csv" % sp), index=False)

    thresholds = {(r["method"], r["seed"]): r.get("threshold", 0.5)
                  for _, r in all_df.iterrows() if not pd.isna(r.get("threshold"))}
    methods = [m for m in METHOD_ORDER if m in set(all_df["method"])]

    pf = per_family_table_multiseed(frames, static_preds, {**art, **tfidf_art},
                                    methods, thresholds, args.seeds)
    pf.to_csv(os.path.join(TABLES, "per_family_errors.csv"), index=False)

    hn = hard_negative_fpr(frames, static_preds, {**art, **tfidf_art}, methods,
                           thresholds, args.seeds)
    hn.to_csv(os.path.join(TABLES, "hard_negative_fpr.csv"), index=False)
    print("\n--- FPR on hard negatives (mean over seeds) ---")
    print(hn.groupby(["split", "method"])["fpr_hard_negatives"].mean()
            .unstack(0).round(4).to_string())

    # Each row says which ablation the proposed model is being compared against,
    # so the table is readable without the calling code.
    sig = pd.concat([
        significance_table(all_df, "M4_CNN_meanpool", "M7_AGNCS_full")
        .assign(comparison="M7 vs M4 (attention at all)"),
        significance_table(all_df, "M5_CNN_attention", "M7_AGNCS_full")
        .assign(comparison="M7 vs M5 (guidance + search head)"),
        significance_table(all_df, "M6_CNN_attn_contrastive", "M7_AGNCS_full")
        .assign(comparison="M7 vs M6 (guidance only)"),
    ], ignore_index=True)
    sig = sig[["comparison"] + [c for c in sig.columns if c != "comparison"]]
    sig.to_csv(os.path.join(TABLES, "significance_tests.csv"), index=False)

    # ------------------------------------------------------------- artefacts
    keep = {k: v for k, v in art.items() if k[0] in ("history", "info")}
    with open(os.path.join(RAW, "histories.pkl"), "wb") as fh:
        pickle.dump(keep, fh)

    # Persist predictions so the figures can be rebuilt without retraining:
    # every probability vector, plus the seed-0 embeddings the retrieval and
    # t-SNE figures need.  Keyed as "<kind>|<method>|<seed>|<split>".
    arrays = {}
    for d in (art, tfidf_art):
        for (kind, meth, seed, sp), v in ((k, v) for k, v in d.items() if len(k) == 4):
            if kind == "prob":
                arrays["prob|%s|%d|%s" % (meth, seed, sp)] = np.asarray(v, dtype=np.float32)
            elif kind == "embed" and seed == args.seeds[0]:
                a = v.toarray() if hasattr(v, "toarray") else v
                arrays["embed|%s|%d|%s" % (meth, seed, sp)] = np.asarray(a, dtype=np.float32)
    for (meth, sp), v in (((m, s), p) for (m, s), p in static_preds.items()):
        arrays["prob|%s|0|%s" % (meth, sp)] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(os.path.join(RAW, "predictions.npz"), **arrays)
    print("  saved %d prediction arrays to %s/predictions.npz" % (len(arrays), RAW))

    # Both attention models, so the guided/unguided comparison figure is
    # reproducible from disk.
    for meth in ("M5_CNN_attention", "M7_AGNCS_full"):
        key = ("model", meth, args.seeds[0])
        if key in art:
            torch.save(art[key].state_dict(),
                       "results/models/%s_seed%d.pt" % (meth, args.seeds[0]))

    # Persist BEFORE plotting: figures are the most fragile stage, and
    # losing an hour of training to a chart bug is not an acceptable
    # failure mode.  With these on disk the figures are rebuildable.
    # ---------------------------------------------------------------- figures
    print("\n[5/5] figures")
    viz.fig_main_comparison(summary)
    # The TF-IDF probabilities live in `tfidf_art`, not `art`; passing only
    # `art` would silently drop that baseline's curve from the figure.
    scored = {**art, **tfidf_art}
    viz.fig_roc_pr(scored, splits,
                   [m for m in methods if m.startswith(("M3", "M4", "M5", "M6", "M7"))],
                   seed=args.seeds[0], static_preds=static_preds)
    viz.fig_training_curves(art, list(VARIANTS), seed=args.seeds[0])
    viz.fig_loss_components(art, "M7_AGNCS_full", seed=args.seeds[0])
    viz.fig_attention_quality(summary)
    viz.fig_retrieval(summary)
    viz.fig_hard_negative_fpr(hn)
    for sp in ("test_id", "test_ood"):
        viz.fig_family_heatmap(pf, sp)

    pairs = []
    for sp in ("test_id", "test_ood"):
        y = splits[sp].y
        p2 = static_preds[("M2_static_taint", sp)] >= 0.5
        pairs.append(("M2 static taint", viz.SPLIT_TITLE[sp], y, p2.astype(int)))
        p7 = art[("prob", "M7_AGNCS_full", args.seeds[0], sp)]
        thr = thresholds.get(("M7_AGNCS_full", args.seeds[0]), 0.5)
        pairs.append(("M7 AGNCS (proposed)", viz.SPLIT_TITLE[sp], y,
                      (p7 >= thr).astype(int)))
    viz.fig_confusion(pairs)

    # Line-level attention: unguided vs guided, on the same function.
    from src.code_tokenizer import parse_rationale
    att_models = {m: art[("model", m, args.seeds[0])]
                  for m in ("M5_CNN_attention", "M7_AGNCS_full")
                  if ("model", m, args.seeds[0]) in art}
    if att_models:
        for sp, fam in (("test_id", "V05_multi_hop_alias"),
                        ("test_id", "S07_allowlisted_identifier"),
                        ("test_ood", "V13_join_fragments")):
            sub = frames[sp][frames[sp].family == fam]
            if not len(sub):
                continue
            r = sub.iloc[0]
            try:
                viz.fig_attention_lines(
                    att_models, r.code, vocab, args.max_len,
                    parse_rationale(r.rationale_lines),
                    title="%s  (%s, %s)" % (fam, "vulnerable" if r.label else "safe", sp),
                    out="fig7_attention_%s.png" % fam)
            except Exception as exc:
                print("  (skipped attention figure for %s: %s)" % (fam, exc))

    try:
        embs = {m: art[("embed", m, args.seeds[0], "test_id")]
                for m in ("M5_CNN_attention", "M7_AGNCS_full")
                if ("embed", m, args.seeds[0], "test_id") in art}
        viz.fig_embedding_space(embs, splits["test_id"].fam, splits["test_id"].y,
                                [], n=2000)
    except Exception as exc:          # t-SNE is optional eye-candy
        print("  (skipped embedding figure: %s)" % exc)

    abl = os.path.join(TABLES, "ablation_train_size.csv")
    if os.path.exists(abl):
        viz.fig_trainsize(pd.read_csv(abl))

    rf_csv = os.path.join(TABLES, "ablation_receptive_field.csv")
    if os.path.exists(rf_csv):
        spans = {}
        for sp in TEST_SPLITS_FIG:
            st = splits[sp]
            spans[sp] = np.array([
                (lambda ix: ix[-1] - ix[0] + 1 if len(ix) else 0)(
                    np.flatnonzero(st.mask[i])) for i in range(len(st))])
        viz.fig_receptive_field(spans, pd.read_csv(rf_csv))

    # Keep the model checkpoint directory present even on a --quick run.
    os.makedirs("results/models", exist_ok=True)

    with open(os.path.join(RAW, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump({**vars(args), "vocab_size": len(vocab),
                   "total_seconds": round(time.time() - t_all, 1)}, fh, indent=2)

    print("\nDONE in %.1f min -- tables in %s, figures in %s"
          % ((time.time() - t_all) / 60, TABLES, FIGS))


if __name__ == "__main__":
    main()
