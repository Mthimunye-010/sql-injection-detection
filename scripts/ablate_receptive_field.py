"""Receptive-field ablation: does the encoder actually span the taint path?

Motivation
----------
A dilated CNN output position sees `1 + 2*sum(dilations)` tokens.  Measuring the
token distance from the first to the last annotated statement in the corpus
shows that the taint path is wide: median 37 tokens in-distribution and 58 in
the held-out families.  A 31-token receptive field therefore covers only about a
third of in-distribution spans and a tenth of the OOD ones -- the encoder
physically cannot connect a source to its sink in most samples.

This script tests the consequence directly.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import prepare, subsample                       # noqa: E402
from src.experiment import eval_retrieval                     # noqa: E402
from src.metrics import attention_metrics, detection_metrics  # noqa: E402
from src.models import receptive_field                        # noqa: E402
from src.train import TrainConfig, predict, train_variant     # noqa: E402

MAX_LEN, D_MODEL, N_TRAIN, EPOCHS = 160, 96, 8000, 5

CONFIGS = [
    (1, 2, 4),              # RF 15
    (1, 2, 4, 8),           # RF 31
    (1, 2, 4, 8, 16),       # RF 63
    (1, 2, 4, 8, 16, 32),   # RF 127
]


def span_stats(splits) -> pd.DataFrame:
    rows = []
    for sp in ("train", "test_id", "test_ood"):
        s = splits[sp]
        spans = []
        for i in range(len(s)):
            idx = np.flatnonzero(s.mask[i])
            spans.append(idx[-1] - idx[0] + 1 if len(idx) else 0)
        spans = np.array(spans)
        row = {"split": sp, "mean": round(float(spans.mean()), 1),
               "median": int(np.median(spans)),
               "p90": int(np.percentile(spans, 90)),
               "p99": int(np.percentile(spans, 99)), "max": int(spans.max())}
        for dil in CONFIGS:
            rf = receptive_field(dil)
            row["cover_rf%d" % rf] = round(float((spans <= rf).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    torch.set_num_threads(6)
    df = pd.read_parquet("data/sqli_corpus.parquet")
    splits, vocab, frames = prepare(df, max_len=MAX_LEN, cache_dir="data/cache")

    os.makedirs("results/tables", exist_ok=True)
    spans = span_stats(splits)
    spans.to_csv("results/tables/taint_span_stats.csv", index=False)
    print("\nTaint-span width vs receptive-field coverage")
    print(spans.to_string(index=False))

    tr = subsample(splits["train"], N_TRAIN, seed=0)
    val = subsample(splits["val"], 2500, seed=0)

    rows = []
    for dil in CONFIGS:
        rf = receptive_field(dil)
        cfg = TrainConfig(epochs=EPOCHS, batch_size=128, seed=0, patience=2,
                          d_model=D_MODEL, dilations=tuple(dil), verbose=False)
        t0 = time.time()
        model, hist, info = train_variant("M7_AGNCS_full", tr, val, len(vocab),
                                          MAX_LEN, cfg)
        secs = time.time() - t0
        row = {"dilations": "-".join(map(str, dil)), "receptive_field": rf,
               "n_params": info["n_params"], "train_seconds": round(secs, 1),
               "best_epoch": info["best_epoch"]}
        for sp in ("test_id", "test_ood"):
            s = splits[sp]
            p = predict(model, s, want_alpha=True)
            dm = detection_metrics(s.y, p["prob"])
            am = attention_metrics(p["alpha"], s.mask, s.valid)
            rm = eval_retrieval(p["embed"], s, seed=0)
            row["%s_f1" % sp] = round(dm["f1"], 4)
            row["%s_fpr" % sp] = round(dm["fpr"], 4)
            row["%s_recall" % sp] = round(dm["recall"], 4)
            row["%s_auc" % sp] = round(dm["roc_auc"], 4)
            row["%s_mrr" % sp] = round(rm["mrr"], 4)
            row["%s_att_lift" % sp] = round(am["att_lift"], 3)
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("results/tables/ablation_receptive_field.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
