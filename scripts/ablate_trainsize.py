"""Training-set-size ablation.

Answers the question a marker will ask: "you have 36,402 training samples, why
did you train on a subset?"  We measure in-distribution and held-out-family
performance as a function of training-set size and show where each saturates.
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

from src.data import prepare, subsample            # noqa: E402
from src.experiment import eval_retrieval                     # noqa: E402
from src.metrics import attention_metrics, detection_metrics  # noqa: E402
from src.train import TrainConfig, predict, train_variant                        # noqa: E402

MAX_LEN, D_MODEL = 160, 96
DILATIONS = (1, 2, 4)   # receptive field 15; see ablate_receptive_field.py
SIZES = [3000, 6000, 12000, 18000, 24000, 36402]


def main() -> None:
    torch.set_num_threads(6)
    df = pd.read_parquet("data/sqli_corpus.parquet")
    splits, vocab, frames = prepare(df, max_len=MAX_LEN, cache_dir="data/cache")
    val = subsample(splits["val"], 2500, seed=0)

    rows = []
    for n in SIZES:
        tr = subsample(splits["train"], n, seed=0)
        cfg = TrainConfig(epochs=5, batch_size=128, seed=0, patience=2,
                          d_model=D_MODEL, dilations=DILATIONS, verbose=False)
        t0 = time.time()
        model, hist, info = train_variant("M7_AGNCS_full", tr, val, len(vocab),
                                          MAX_LEN, cfg)
        secs = time.time() - t0
        row = {"n_train": len(tr), "epochs_run": len(hist["epoch"]),
               "best_epoch": info["best_epoch"], "train_seconds": round(secs, 1)}
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
    os.makedirs("results/tables", exist_ok=True)
    out.to_csv("results/tables/ablation_train_size.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
