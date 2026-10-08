"""Evaluation metrics: detection, code search (retrieval), and attention quality.

Every formula reported in the write-up is implemented here, with the notation
used in the report kept in the docstrings so the two can be checked against
each other.

Notation
--------
TP, FP, TN, FN   entries of the confusion matrix at decision threshold tau
N                number of evaluated samples
y_i in {0,1}     ground-truth label of sample i
p_i in [0,1]     predicted probability of the positive (vulnerable) class
alpha_{i,t}      attention weight of model on token t of sample i
m_{i,t} in {0,1} ground-truth taint/evidence annotation of token t
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def confusion(y: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    y = np.asarray(y).astype(int)
    p = np.asarray(pred).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def detection_metrics(y: np.ndarray, prob: np.ndarray, thr: float = 0.5
                      ) -> Dict[str, float]:
    """Full detection report.

        Accuracy  = (TP + TN) / N
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)                     [= TPR, sensitivity]
        F1        = 2 * Precision * Recall / (Precision + Recall)
        FPR       = FP / (FP + TN)                     [the headline metric]
        Specificity = TN / (TN + FP) = 1 - FPR
        BalancedAcc = (Recall + Specificity) / 2
        MCC       = (TP*TN - FP*FN) /
                    sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    FPR is singled out because the project's premise is that static analysers
    are *too noisy to use*, not that they miss too much: an analyst's review
    budget is spent on false positives.
    """
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y).astype(int)
    pred = (prob >= thr).astype(int)
    tp, fp, tn, fn = confusion(y, pred)
    n = tp + fp + tn + fn

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    spec = 1.0 - fpr
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    out = {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (rec + spec),
        "mcc": mcc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    out["roc_auc"] = roc_auc(y, prob)
    out["pr_auc"] = average_precision(y, prob)
    return out


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC-AUC (equals the Mann-Whitney U statistic, ties averaged)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def best_threshold(y: np.ndarray, prob: np.ndarray,
                   objective: str = "f1") -> Tuple[float, float]:
    """Threshold selected on a *validation* split only.

    `objective="f1"` maximises F1.  `objective="fpr_at_recall"` picks the
    threshold that minimises FPR subject to recall >= 0.95, which is how a
    security team would actually tune a detector.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    cands = np.unique(np.concatenate([[0.0, 1.0], np.quantile(prob, np.linspace(0, 1, 201))]))
    best_t, best_v = 0.5, -1.0
    for t in cands:
        m = detection_metrics(y, prob, thr=float(t))
        if objective == "f1":
            v = m["f1"]
        elif objective == "mcc":
            v = m["mcc"]
        else:
            v = (1.0 - m["fpr"]) if m["recall"] >= 0.95 else -1.0
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t, best_v


# --------------------------------------------------------------------------- #
# Code search (retrieval)
# --------------------------------------------------------------------------- #

def retrieval_metrics(query_emb: np.ndarray, corpus_emb: np.ndarray,
                      query_fam: np.ndarray, corpus_fam: np.ndarray,
                      ks: Sequence[int] = (1, 5, 10),
                      exclude_self: Optional[np.ndarray] = None
                      ) -> Dict[str, float]:
    """Query-by-example code search evaluated with family-level relevance.

    A corpus item is *relevant* to a query if it belongs to the same
    vulnerability family -- i.e. it realises the same taint pattern.  Both
    embedding matrices are L2-normalised, so the cosine similarity is a dot
    product and ranking is a single matrix multiply.

        MRR       = (1/|Q|) sum_q 1 / rank of the first relevant hit
        Recall@k  = (1/|Q|) sum_q 1[ a relevant item appears in the top k ]
        P@k       = (1/|Q|) sum_q (relevant items in top k) / k
        MAP       = (1/|Q|) sum_q (1/R_q) sum_{j: rel} P@j
        nDCG@k    = DCG@k / IDCG@k,
                    DCG@k = sum_{j=1..k} rel_j / log2(j + 1)

    `exclude_self` holds, for each query, the corpus row index to mask out when
    the query itself is a member of the corpus.
    """
    qe = np.asarray(query_emb, dtype=np.float32)
    ce = np.asarray(corpus_emb, dtype=np.float32)
    sim = qe @ ce.T                                        # (|Q|, |C|)
    if exclude_self is not None:
        sim[np.arange(len(qe)), np.asarray(exclude_self)] = -np.inf

    rel_all = (np.asarray(query_fam)[:, None] == np.asarray(corpus_fam)[None, :])
    if exclude_self is not None:
        rel_all[np.arange(len(qe)), np.asarray(exclude_self)] = False

    kmax = max(ks)
    # Keep the temporary index matrix bounded.  argpartition allocates an
    # int-sized array the shape of its input, which is large even for a modest
    # query batch when the corpus contains thousands of functions.
    top = np.empty((len(qe), kmax), dtype=np.int64)
    top_chunk = 64
    for a in range(0, len(qe), top_chunk):
        b = min(len(qe), a + top_chunk)
        part = np.argpartition(-sim[a:b],
                               kth=min(kmax, sim.shape[1] - 1), axis=1)[:, :kmax]
        row_chunk = np.arange(b - a)[:, None]
        top[a:b] = part[row_chunk,
                        np.argsort(-sim[a:b][row_chunk, part], axis=1)]
    row = np.arange(len(qe))[:, None]
    rel_top = rel_all[row, top]                            # (|Q|, kmax)

    out: Dict[str, float] = {}
    for k in ks:
        out["recall@%d" % k] = float(rel_top[:, :k].any(axis=1).mean())
        out["p@%d" % k] = float(rel_top[:, :k].mean())

    # MRR and nDCG over the top-kmax window (first hit is essentially always in it)
    first = np.argmax(rel_top, axis=1).astype(float)
    hit = rel_top.any(axis=1)
    rr = np.where(hit, 1.0 / (first + 1.0), 0.0)
    out["mrr"] = float(rr.mean())

    disc = 1.0 / np.log2(np.arange(1, kmax + 1) + 1.0)
    dcg = (rel_top * disc).sum(axis=1)
    n_rel = rel_all.sum(axis=1)
    idcg = np.array([disc[:min(int(r), kmax)].sum() for r in n_rel])
    with np.errstate(invalid="ignore", divide="ignore"):
        ndcg = np.where(idcg > 0, dcg / idcg, 0.0)
    out["ndcg@%d" % kmax] = float(np.nanmean(ndcg))

    # Mean average precision over the full ranking (exact, not truncated).
    # Computed in query chunks: the dense (|Q| x |C|) rank matrix is the one
    # array here that can exceed memory on a large corpus.
    nq, nc = sim.shape
    ranks_row = np.arange(1, nc + 1, dtype=np.float32)
    ap = np.zeros(nq, dtype=np.float64)
    chunk = min(64, max(1, int(4e7 // max(nc, 1))))
    for a in range(0, nq, chunk):
        b = min(nq, a + chunk)
        order = np.argsort(-sim[a:b], axis=1)
        rel_sorted = rel_all[np.arange(a, b)[:, None], order]
        csum = np.cumsum(rel_sorted, axis=1, dtype=np.float32)
        prec_at = csum / ranks_row[None, :]
        ap[a:b] = (prec_at * rel_sorted).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = np.where(n_rel > 0, ap / np.maximum(n_rel, 1), 0.0)
    out["map"] = float(np.nanmean(ap))
    return out


# --------------------------------------------------------------------------- #
# Attention quality -- does the model look where the vulnerability is?
# --------------------------------------------------------------------------- #

def attention_metrics(alpha: np.ndarray, mask: np.ndarray, valid: np.ndarray
                      ) -> Dict[str, float]:
    """How well the attention map agrees with the ground-truth annotation.

        AMR_i (attention mass on rationale) = sum_t alpha_{i,t} m_{i,t}
        Lift_i   = AMR_i / (|m_i| / |valid_i|)
                   -- AMR divided by the mass a uniform map would place there,
                      so Lift = 1 means "no better than looking everywhere".
        Pointing_i = 1[ argmax_t alpha_{i,t} is annotated ]
        IoU@|m|_i  = |top-|m_i| attention tokens  ∩  annotated| / |union|

    Only samples with a non-empty annotation are scored.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    valid = np.asarray(valid).astype(bool)

    keep = (mask & valid).any(axis=1)
    if not keep.any():
        return {"att_mass_rationale": float("nan"), "att_lift": float("nan"),
                "att_pointing": float("nan"), "att_iou": float("nan"), "n": 0}

    a = alpha[keep].copy()
    m = mask[keep] & valid[keep]
    v = valid[keep]
    a[~v] = 0.0
    a = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)

    amr = (a * m).sum(axis=1)
    base = m.sum(axis=1) / np.maximum(v.sum(axis=1), 1)
    lift = amr / np.maximum(base, 1e-12)

    a_pick = np.where(v, a, -np.inf)
    pointing = m[np.arange(len(a)), np.argmax(a_pick, axis=1)]

    ious = np.empty(len(a))
    for i in range(len(a)):
        k = int(m[i].sum())
        idx = np.argsort(-a_pick[i])[:k]
        sel = np.zeros(a.shape[1], dtype=bool)
        sel[idx] = True
        inter = (sel & m[i]).sum()
        union = (sel | m[i]).sum()
        ious[i] = inter / union if union else 0.0

    return {
        "att_mass_rationale": float(amr.mean()),
        "att_lift": float(lift.mean()),
        "att_pointing": float(pointing.mean()),
        "att_iou": float(ious.mean()),
        "n": int(keep.sum()),
    }


# --------------------------------------------------------------------------- #
# Significance testing across seeds
# --------------------------------------------------------------------------- #

def paired_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison of two methods across matched seeds.

    Reports the mean difference, a paired t-test, the Wilcoxon signed-rank test
    (no normality assumption), and Cohen's d_z for paired samples.
    """
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    out = {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(d.mean()), "n": int(len(d))}
    if len(d) < 2 or np.allclose(d, 0):
        out.update({"t_stat": float("nan"), "t_p": float("nan"),
                    "wilcoxon_p": float("nan"), "cohen_dz": 0.0})
        return out
    t, p = stats.ttest_rel(a, b)
    out["t_stat"], out["t_p"] = float(t), float(p)
    try:
        _, wp = stats.wilcoxon(a, b)
        out["wilcoxon_p"] = float(wp)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    sd = d.std(ddof=1)
    out["cohen_dz"] = float(d.mean() / sd) if sd > 0 else float("inf")
    return out


def mean_std(values: Sequence[float]) -> str:
    v = np.asarray(values, dtype=float)
    return "%.4f +/- %.4f" % (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
