"""Figures for the report.

House style: one colour per method held constant across every figure, an
Okabe-Ito colour-blind-safe palette, no gridlines competing with the data, and
direct value labels wherever a reader would otherwise have to measure a bar
against an axis.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .code_tokenizer import parse_rationale, rationale_mask, tokenise

# Okabe-Ito, colour-blind safe.  The proposed model is the dark navy so it
# stays distinguishable in greyscale print.
METHOD_COLOR = {
    "M1_static_regex": "#D55E00",
    "M2_static_taint": "#E69F00",
    "M3_tfidf_lr": "#CC79A7",
    "M4_CNN_meanpool": "#56B4E9",
    "M5_CNN_attention": "#0072B2",
    "M6_CNN_attn_contrastive": "#009E73",
    "M7_AGNCS_full": "#1B2A4A",
}
METHOD_SHORT = {
    "M1_static_regex": "M1 static\n(patterns)",
    "M2_static_taint": "M2 static\n(taint)",
    "M3_tfidf_lr": "M3 TF-IDF\n+ LR",
    "M4_CNN_meanpool": "M4 CNN\nmean-pool",
    "M5_CNN_attention": "M5 CNN\n+ attn",
    "M6_CNN_attn_contrastive": "M6 + contrast",
    "M7_AGNCS_full": "M7 AGNCS\n(proposed)",
}
SPLIT_TITLE = {"test_id": "In-distribution families",
               "test_ood": "Held-out families (OOD)"}

INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#DCDCDC"


def set_style() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig: plt.Figure, name: str, out_dir: str = "results/figures") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    return path


def _methods_in(df: pd.DataFrame) -> List[str]:
    order = list(METHOD_COLOR)
    return [m for m in order if m in set(df["method"])]


# --------------------------------------------------------------------------- #
# Figure 1 -- headline comparison
# --------------------------------------------------------------------------- #

def fig_main_comparison(summary: pd.DataFrame,
                        metrics: Sequence[str] = ("f1", "fpr"),
                        out: str = "fig1_detection_comparison.png") -> plt.Figure:
    splits = ["test_id", "test_ood"]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(10.5, 3.1 * len(metrics)),
                             sharey="row")
    axes = np.atleast_2d(axes)
    for r, metric in enumerate(metrics):
        for c, sp in enumerate(splits):
            ax = axes[r, c]
            sub = summary[summary["split"] == sp]
            meths = _methods_in(sub)
            vals = [float(sub.loc[sub.method == m, metric].iloc[0]) for m in meths]
            errs = [float(sub.loc[sub.method == m, metric + "_sd"].iloc[0])
                    if metric + "_sd" in sub.columns else 0.0 for m in meths]
            x = np.arange(len(meths))
            bars = ax.bar(x, vals, yerr=errs, capsize=3,
                          color=[METHOD_COLOR[m] for m in meths],
                          edgecolor="white", linewidth=0.8, width=0.72)
            for xi, v, e in zip(x, vals, errs):
                if np.isnan(v):
                    continue
                ax.text(xi, v + e + 0.018, "%.3f" % v, ha="center", va="bottom",
                        fontsize=8, color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_SHORT[m] for m in meths], fontsize=7.6)
            ax.set_ylim(0, max(1.0, max([v for v in vals if not np.isnan(v)] + [0]) * 1.25))
            if c == 0:
                ax.set_ylabel({"f1": "F1 score",
                               "fpr": "False-positive rate"}.get(metric, metric))
            if r == 0:
                ax.set_title(SPLIT_TITLE[sp], pad=8)
            if metric == "fpr":
                ax.set_ylim(0, max(0.1, np.nanmax(vals) * 1.3))
    fig.suptitle("Detection performance: higher F1 is better, lower FPR is better",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 -- ROC and precision-recall curves
# --------------------------------------------------------------------------- #

def _roc_points(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-s)
    y = np.asarray(y)[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / max(1, (y == 1).sum())
    fpr = fp / max(1, (y == 0).sum())
    return np.r_[0, fpr], np.r_[0, tpr]


def _pr_points(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-s)
    y = np.asarray(y)[order]
    tp = np.cumsum(y == 1)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(1, (y == 1).sum())
    return rec, prec


def fig_roc_pr(artefacts: dict, splits: dict, methods: Sequence[str],
               seed: int = 0, static_preds: Optional[dict] = None,
               out: str = "fig2_roc_pr.png") -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6))
    for c, sp in enumerate(["test_id", "test_ood"]):
        y = splits[sp].y
        ax_roc, ax_pr = axes[0, c], axes[1, c]
        for m in methods:
            p = artefacts.get(("prob", m, seed, sp))
            if p is None:
                continue
            f, t = _roc_points(y, p)
            ax_roc.plot(f, t, color=METHOD_COLOR[m], lw=1.9,
                        label=METHOD_SHORT[m].replace("\n", " "),
                        zorder=3 if m == "M7_AGNCS_full" else 2)
            r, pr = _pr_points(y, p)
            ax_pr.plot(r, pr, color=METHOD_COLOR[m], lw=1.9,
                       zorder=3 if m == "M7_AGNCS_full" else 2)
        # Static analysers are single operating points, not curves.
        if static_preds:
            for m in ("M1_static_regex", "M2_static_taint"):
                p = static_preds.get((m, sp))
                if p is None:
                    continue
                pred = (p >= 0.5)
                tpr = float((pred & (y == 1)).sum() / max(1, (y == 1).sum()))
                fpr = float((pred & (y == 0)).sum() / max(1, (y == 0).sum()))
                prec = float((pred & (y == 1)).sum() / max(1, pred.sum()))
                ax_roc.scatter([fpr], [tpr], s=58, marker="D",
                               color=METHOD_COLOR[m], zorder=5,
                               edgecolor="white", linewidth=1.0,
                               label=METHOD_SHORT[m].replace("\n", " "))
                ax_pr.scatter([tpr], [prec], s=58, marker="D",
                              color=METHOD_COLOR[m], zorder=5,
                              edgecolor="white", linewidth=1.0)
        ax_roc.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
        ax_roc.set_title("ROC -- %s" % SPLIT_TITLE[sp])
        ax_roc.set_xlabel("False-positive rate")
        ax_roc.set_ylabel("True-positive rate")
        ax_pr.set_title("Precision-recall -- %s" % SPLIT_TITLE[sp])
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        for a in (ax_roc, ax_pr):
            a.set_xlim(-0.02, 1.02)
            a.set_ylim(-0.02, 1.04)
    axes[0, 0].legend(fontsize=7.6, loc="lower right")
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 -- training dynamics
# --------------------------------------------------------------------------- #

def fig_training_curves(artefacts: dict, variants: Sequence[str], seed: int = 0,
                        out: str = "fig3_training_curves.png") -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.3))
    for v in variants:
        h = artefacts.get(("history", v, seed))
        if not h:
            continue
        ep = h["epoch"]
        axes[0].plot(ep, h["train_loss"], color=METHOD_COLOR[v], lw=1.8,
                     marker="o", ms=3.2, label=METHOD_SHORT[v].replace("\n", " "))
        axes[1].plot(ep, h["val_f1"], color=METHOD_COLOR[v], lw=1.8,
                     marker="o", ms=3.2)
        axes[2].plot(ep, h["val_fpr"], color=METHOD_COLOR[v], lw=1.8,
                     marker="o", ms=3.2)
    axes[0].set_title("Total training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[2].set_title("Validation false-positive rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("FPR")
    axes[0].legend(fontsize=7.8)
    fig.tight_layout()
    save(fig, out)
    return fig


def fig_loss_components(artefacts: dict, variant: str = "M7_AGNCS_full",
                        seed: int = 0,
                        out: str = "fig3b_loss_components.png") -> plt.Figure:
    h = artefacts.get(("history", variant, seed))
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    if h:
        ep = h["epoch"]
        ax.plot(ep, h["cls"], lw=1.9, marker="o", ms=3.4, color="#0072B2",
                label=r"$\mathcal{L}_{cls}$ (detection)")
        key = "att_kl" if h.get("att_kl") else "att"
        ax.plot(ep, h[key], lw=1.9, marker="s", ms=3.4, color="#D55E00",
                label=r"$KL(q\,\|\,\alpha)$ (attention guidance)")
        ax.plot(ep, h["ret"], lw=1.9, marker="^", ms=3.4, color="#009E73",
                label=r"$\mathcal{L}_{ret}$ (contrastive search)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component")
    ax.set_title("AGNCS loss decomposition")
    ax.text(0.98, 0.96, "guidance shown as KL: 0 = attention matches "
            "the annotation exactly",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 -- confusion matrices
# --------------------------------------------------------------------------- #

def fig_confusion(pairs: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
                  out: str = "fig4_confusion.png") -> plt.Figure:
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(2.85 * n, 3.0))
    axes = np.atleast_1d(axes)
    for ax, (title, sub, y, pred) in zip(axes, pairs):
        cm = np.zeros((2, 2), dtype=int)
        for a in (0, 1):
            for b in (0, 1):
                cm[a, b] = int(((y == a) & (pred == b)).sum())
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for a in (0, 1):
            for b in (0, 1):
                ax.text(b, a, "%d\n(%.1f%%)" % (cm[a, b], 100 * norm[a, b]),
                        ha="center", va="center", fontsize=8.6,
                        color="white" if norm[a, b] > 0.55 else INK)
        ax.set_xticks([0, 1], ["pred safe", "pred vuln"], fontsize=8)
        ax.set_yticks([0, 1], ["safe", "vulnerable"], fontsize=8)
        ax.set_title("%s\n%s" % (title, sub), fontsize=9)
        ax.grid(False)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 5 -- per-family error heatmap
# --------------------------------------------------------------------------- #

def fig_family_heatmap(per_family: pd.DataFrame, split: str,
                       out: Optional[str] = None) -> plt.Figure:
    sub = per_family[per_family["split"] == split]
    meths = [m for m in METHOD_COLOR if m in set(sub["method"])]
    fams = sorted(sub["family"].unique())
    M = np.full((len(meths), len(fams)), np.nan)
    for i, m in enumerate(meths):
        d = sub[sub.method == m].set_index("family")
        for j, f in enumerate(fams):
            if f in d.index:
                M[i, j] = d.loc[f, "error_rate"]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.42 * len(fams) + 3), 0.46 * len(meths) + 2.2))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    labels = sub.groupby("family")["label"].first()
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(["%s %s" % ("[V]" if labels[f] else "[S]", f.split("_", 1)[1])
                        for f in fams], rotation=60, ha="right", fontsize=7.2)
    ax.set_yticks(range(len(meths)))
    ax.set_yticklabels([METHOD_SHORT[m].replace("\n", " ") for m in meths], fontsize=8)
    for i in range(len(meths)):
        for j in range(len(fams)):
            if not np.isnan(M[i, j]) and M[i, j] >= 0.10:
                ax.text(j, i, "%.0f" % (100 * M[i, j]), ha="center", va="center",
                        fontsize=6.4, color="white" if M[i, j] > 0.55 else INK)
    ax.set_title("Per-family error rate (%%), %s  --  [V] vulnerable, [S] safe"
                 % SPLIT_TITLE[split], fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.016, pad=0.012, label="error rate")
    fig.tight_layout()
    save(fig, out or ("fig5_family_errors_%s.png" % split))
    return fig


# --------------------------------------------------------------------------- #
# Figure 6 -- attention quality
# --------------------------------------------------------------------------- #

def fig_attention_quality(summary: pd.DataFrame,
                          out: str = "fig6_attention_quality.png") -> plt.Figure:
    metrics = [("att_mass_rationale", "Attention mass on\nannotated statements"),
               ("att_lift", "Lift over uniform\nattention"),
               ("att_pointing", "Pointing-game\naccuracy"),
               ("att_iou", "Top-k attention\nIoU")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.3))
    splits = ["test_id", "test_ood"]
    for ax, (key, title) in zip(axes, metrics):
        sub = summary[summary[key].notna()] if key in summary.columns else summary.iloc[0:0]
        meths = [m for m in METHOD_COLOR if m in set(sub["method"])]
        w = 0.38
        x = np.arange(len(meths))
        for si, sp in enumerate(splits):
            s2 = sub[sub.split == sp]
            vals, errs = [], []
            for m in meths:
                r = s2[s2.method == m]
                vals.append(float(r[key].iloc[0]) if len(r) else np.nan)
                errs.append(float(r[key + "_sd"].iloc[0])
                            if len(r) and key + "_sd" in r else 0.0)
            ax.bar(x + (si - 0.5) * w, vals, w, yerr=errs, capsize=2.5,
                   color=[METHOD_COLOR[m] for m in meths],
                   alpha=1.0 if si == 0 else 0.55, edgecolor="white",
                   linewidth=0.7, label=SPLIT_TITLE[sp] if ax is axes[0] else None)
        if key == "att_lift":
            ax.axhline(1.0, ls="--", lw=1, color=MUTED)
            ax.text(len(meths) - 0.5, 1.03, "uniform", fontsize=7, color=MUTED,
                    ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in meths], fontsize=7.2)
        ax.set_title(title, fontsize=9)
    if len(axes):
        axes[0].legend(fontsize=7.4, loc="upper left")
    fig.suptitle("Does the model look where the vulnerability actually is?",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 7 -- line-level attention on a real snippet
# --------------------------------------------------------------------------- #

def line_attention(model, code: str, vocab, max_len: int) -> Tuple[List[str], np.ndarray]:
    """Per-line attention mass for one snippet."""
    import torch
    toks, lines = tokenise(code)
    toks, lines = toks[:max_len], lines[:max_len]
    ids = vocab.encode(toks)
    x = np.full((1, max_len), vocab.pad_id, dtype=np.int64)
    x[0, :len(ids)] = ids
    with torch.no_grad():
        out = model(torch.from_numpy(x))
    a = out["alpha"][0].numpy()[:len(ids)]
    src = code.split("\n")
    per = np.zeros(len(src))
    for w, ln in zip(a, lines):
        if 0 <= ln < len(per):
            per[ln] += float(w)
    return src, per


def fig_attention_lines(models: Dict[str, object], code: str, vocab, max_len: int,
                        rationale: Sequence[int], title: str = "",
                        out: str = "fig7_attention_lines.png") -> plt.Figure:
    names = list(models)
    per = {n: line_attention(models[n], code, vocab, max_len)[1] for n in names}
    src = code.split("\n")
    n_lines = len(src)
    fig, axes = plt.subplots(
        1, len(names), figsize=(6.4 + 2.5 * len(names), max(3.0, 0.30 * n_lines)),
        sharey=True)
    axes = np.atleast_1d(axes)
    rset = set(rationale)
    for ax, n in zip(axes, names):
        v = per[n]
        v = v / max(v.sum(), 1e-9)
        colors = ["#D55E00" if i in rset else "#9FB3C8" for i in range(n_lines)]
        ax.barh(np.arange(n_lines), v, color=colors, height=0.72)
        ax.invert_yaxis()
        ax.set_title(METHOD_SHORT.get(n, n).replace("\n", " "), fontsize=9)
        ax.set_xlabel("attention mass")
        ax.set_xlim(0, max(0.12, max(per[k].max() / max(per[k].sum(), 1e-9)
                                     for k in names) * 1.15))
    axes[0].set_yticks(np.arange(n_lines))
    axes[0].set_yticklabels(
        [("%2d  %s" % (i, s[:70].rstrip())) if s.strip() else "%2d" % i
         for i, s in enumerate(src)],
        fontsize=7.0, fontfamily="monospace")
    for lbl, i in zip(axes[0].get_yticklabels(), range(n_lines)):
        if i in rset:
            lbl.set_color("#B3400A")
            lbl.set_fontweight("bold")
    fig.suptitle(title or "Line-level attention (orange = annotated taint path)",
                 y=1.01, fontsize=10.5)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 8 -- code-search quality
# --------------------------------------------------------------------------- #

def fig_retrieval(summary: pd.DataFrame,
                  metrics: Sequence[str] = ("mrr", "recall@1", "recall@10", "map"),
                  out: str = "fig8_retrieval.png") -> plt.Figure:
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.9 * len(metrics), 3.3))
    axes = np.atleast_1d(axes)
    splits = ["test_id", "test_ood"]
    for ax, key in zip(axes, metrics):
        sub = summary[summary[key].notna()] if key in summary.columns else summary.iloc[0:0]
        meths = [m for m in METHOD_COLOR if m in set(sub["method"])]
        x = np.arange(len(meths))
        w = 0.38
        for si, sp in enumerate(splits):
            s2 = sub[sub.split == sp]
            vals = [float(s2.loc[s2.method == m, key].iloc[0])
                    if len(s2[s2.method == m]) else np.nan for m in meths]
            errs = [float(s2.loc[s2.method == m, key + "_sd"].iloc[0])
                    if len(s2[s2.method == m]) and key + "_sd" in s2 else 0.0
                    for m in meths]
            ax.bar(x + (si - 0.5) * w, vals, w, yerr=errs, capsize=2.5,
                   color=[METHOD_COLOR[m] for m in meths],
                   alpha=1.0 if si == 0 else 0.55, edgecolor="white", linewidth=0.7,
                   label=SPLIT_TITLE[sp] if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in meths], fontsize=7.2)
        ax.set_ylim(0, 1.05)
        ax.set_title(key.upper(), fontsize=9.5)
    axes[0].legend(fontsize=7.4, loc="lower left")
    fig.suptitle("Neural code search: retrieving code that realises the same "
                 "vulnerability pattern", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 9 -- the learned search space
# --------------------------------------------------------------------------- #

def fig_embedding_space(embeds: Dict[str, np.ndarray], fam: np.ndarray,
                        labels: np.ndarray, fam_names: Sequence[str],
                        n: int = 2000, seed: int = 0,
                        out: str = "fig9_embedding_space.png") -> plt.Figure:
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    names = list(embeds)
    idx = rng.permutation(len(fam))[:n]
    fig, axes = plt.subplots(1, len(names), figsize=(5.1 * len(names), 4.8))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab20")
    for ax, nm in zip(axes, names):
        Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed,
                 max_iter=600).fit_transform(embeds[nm][idx].astype(np.float32))
        f = fam[idx]
        for k, fid in enumerate(np.unique(f)):
            s = f == fid
            ax.scatter(Z[s, 0], Z[s, 1], s=5.5, alpha=0.72, color=cmap(k % 20),
                       linewidths=0)
        vuln = labels[idx] == 1
        ax.scatter([], [], c="none")
        ax.set_title("%s\n(%d points, colour = vulnerability family)"
                     % (METHOD_SHORT.get(nm, nm).replace("\n", " "), len(idx)),
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.text(0.02, 0.02, "%.0f%% vulnerable" % (100 * vuln.mean()),
                transform=ax.transAxes, fontsize=7.6, color=MUTED)
    fig.suptitle("t-SNE of the code-search embedding space", y=1.0, fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 10 -- training-size ablation
# --------------------------------------------------------------------------- #

def fig_trainsize(df: pd.DataFrame,
                  out: str = "fig10_trainsize_ablation.png") -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    specs = [("f1", "F1 score"), ("fpr", "False-positive rate"),
             ("mrr", "Search MRR")]
    for ax, (key, title) in zip(axes, specs):
        for sp, color, mark in (("test_id", "#0072B2", "o"),
                                ("test_ood", "#D55E00", "s")):
            col = "%s_%s" % (sp, key)
            if col not in df.columns:
                continue
            ax.plot(df["n_train"], df[col], color=color, marker=mark, ms=4.2,
                    lw=1.9, label=SPLIT_TITLE[sp])
        ax.set_xscale("log")
        ax.set_xlabel("Training samples")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(df["n_train"])
        ax.set_xticklabels(["%dk" % round(v / 1000) if v >= 1000 else str(v)
                            for v in df["n_train"]], fontsize=7.6)
        ax.minorticks_off()
    axes[0].legend(fontsize=7.8)
    fig.suptitle("Where does performance saturate? (AGNCS, seed 0)", y=1.02,
                 fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 11 -- the metric that decides deployability
# --------------------------------------------------------------------------- #

def fig_hard_negative_fpr(hn: pd.DataFrame,
                          out: str = "fig11_hard_negative_fpr.png") -> plt.Figure:
    g = (hn.groupby(["split", "method"])[["fpr_hard_negatives", "fpr_easy_negatives"]]
           .agg(["mean", "std"]).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), sharey=True)
    for ax, sp in zip(axes, ["test_id", "test_ood"]):
        sub = g[g["split"] == sp]
        meths = [m for m in METHOD_COLOR if m in set(sub["method"])]
        x = np.arange(len(meths))
        w = 0.38
        hard = [float(sub.loc[sub.method == m, ("fpr_hard_negatives", "mean")].iloc[0])
                for m in meths]
        herr = [float(np.nan_to_num(sub.loc[sub.method == m, ("fpr_hard_negatives", "std")].iloc[0]))
                for m in meths]
        easy = [float(np.nan_to_num(sub.loc[sub.method == m, ("fpr_easy_negatives", "mean")].iloc[0]))
                for m in meths]
        ax.bar(x - w / 2, hard, w, yerr=herr, capsize=2.5,
               color=[METHOD_COLOR[m] for m in meths], edgecolor="white",
               linewidth=0.7, label="hard negatives (look dangerous, are safe)")
        ax.bar(x + w / 2, easy, w, color=[METHOD_COLOR[m] for m in meths],
               alpha=0.4, edgecolor="white", linewidth=0.7,
               label="ordinary safe code")
        for xi, v in zip(x, hard):
            if not np.isnan(v):
                ax.text(xi - w / 2, v + 0.012, "%.2f" % v, ha="center",
                        va="bottom", fontsize=7.2)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in meths], fontsize=7.2)
        ax.set_title(SPLIT_TITLE[sp], fontsize=9.5)
    axes[0].set_ylabel("False-positive rate")
    axes[0].legend(fontsize=7.4)
    fig.suptitle("False positives where they hurt: safe code that looks dangerous",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Figure 12 -- taint span vs receptive field (the refuted hypothesis)
# --------------------------------------------------------------------------- #

def fig_receptive_field(spans: Dict[str, np.ndarray], abl: pd.DataFrame,
                        candidates: Sequence[int] = (15, 31, 63, 127),
                        out: str = "fig12_receptive_field.png") -> plt.Figure:
    """Left: how far apart the annotated statements are, against the candidate
    receptive fields.  Right: what actually happened when the field was widened.

    The pair is the point: the left panel motivates a wide field, the right
    panel shows the motivation was wrong.
    """
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.0, 3.6))

    for name, color in (("test_id", "#0072B2"), ("test_ood", "#D55E00")):
        if name not in spans:
            continue
        v = np.sort(np.asarray(spans[name]))
        axl.plot(v, np.linspace(0, 1, len(v)), color=color, lw=2.0,
                 label=SPLIT_TITLE[name])
    for r in candidates:
        axl.axvline(r, color=MUTED, ls=":", lw=1.0)
        axl.text(r, 1.02, "R=%d" % r, rotation=0, fontsize=6.8, color=MUTED,
                 ha="center", va="bottom")
    axl.set_xlabel("Taint span (tokens from first to last annotated token)")
    axl.set_ylabel("Cumulative fraction of samples")
    axl.set_title("The spans are wide -- which predicts a wide field is needed",
                  fontsize=9.5)
    axl.set_xlim(0, 130)
    axl.set_ylim(0, 1.06)
    axl.legend(fontsize=7.8, loc="lower right")

    specs = [("test_id_f1", "In-distribution F1", "#0072B2", "o", "-"),
             ("test_ood_f1", "Held-out F1", "#D55E00", "s", "-"),
             ("test_ood_auc", "Held-out ROC-AUC", "#009E73", "^", "--"),
             ("test_ood_fpr", "Held-out FPR", "#CC79A7", "v", "--")]
    for key, label, color, mark, ls in specs:
        if key not in abl.columns:
            continue
        axr.plot(abl["receptive_field"], abl[key], color=color, marker=mark,
                 ms=5.0, lw=1.9, ls=ls, label=label)
    axr.set_xscale("log", base=2)
    axr.set_xticks(list(abl["receptive_field"]))
    axr.set_xticklabels([str(int(v)) for v in abl["receptive_field"]])
    axr.minorticks_off()
    axr.set_xlabel(r"Receptive field $R = 1 + 2\sum_i d_i$ (tokens)")
    axr.set_ylabel("Score")
    axr.set_ylim(0, 1.05)
    axr.set_title("...but widening it does not help (1 seed; see text)",
                  fontsize=9.5)
    axr.legend(fontsize=7.4, loc="center left")
    fig.suptitle("Taint span and receptive field: a prediction and its refutation",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, out)
    return fig
