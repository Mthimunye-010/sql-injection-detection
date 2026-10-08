"""Generate the large synthetic SQL-injection corpus and its evaluation splits.

Usage
-----
    python scripts/generate_dataset.py                 # default: 60,000 samples
    python scripts/generate_dataset.py --n-total 20000 # smaller, faster

Splits
------
24 of the 30 families form the **in-distribution (ID)** pool and are split
70/10/20 into train / validation / test-ID.

6 families are held out *entirely* -- they never appear in training -- and form
the **out-of-distribution (OOD)** test set.  Reporting ID and OOD separately is
what turns the experiment from "can the model fit this grammar?" into "does the
learned notion of a taint flow transfer to vulnerability patterns it has never
seen?", which is the claim the project brief actually makes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corpus_builder import balanced_plan, generate_corpus          # noqa: E402
from src.sqli_corpus import (ALL_FAMILIES, HARD_NEGATIVE_FAMILIES,      # noqa: E402
                             SAFE_FAMILIES, VULN_FAMILIES)

# Families withheld from training entirely.  Chosen to be *hard* on both sides:
# two inter-procedural / fragment-assembly vulnerabilities, one template-engine
# vulnerability, and three safe patterns that look dangerous on the surface.
OOD_FAMILIES = [
    "V06_helper_builder",
    "V11_template_substitute",
    "V13_join_fragments",
    "S08_enum_fragment_lookup",
    "S12_sql_string_not_executed",
    "S15_composable_identifier_api",
]
ID_FAMILIES = [f for f in ALL_FAMILIES if f not in OOD_FAMILIES]

TRAIN_FRAC, VAL_FRAC = 0.70, 0.10


def stratified_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign train/val/test_id within every family, so each split keeps the
    exact family and class composition of the ID pool."""
    df = df.copy()
    df["split"] = "test_id"
    for fam, idx in df.groupby("family").groups.items():
        idx = pd.Series(list(idx)).sample(frac=1.0, random_state=seed).tolist()
        n = len(idx)
        n_tr = int(round(TRAIN_FRAC * n))
        n_va = int(round(VAL_FRAC * n))
        df.loc[idx[:n_tr], "split"] = "train"
        df.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=60000)
    ap.add_argument("--ood-frac", type=float, default=0.1333,
                    help="share of the corpus drawn from held-out families")
    ap.add_argument("--seed", type=int, default=20260404)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n_ood = int(round(args.n_total * args.ood_frac))
    n_id = args.n_total - n_ood

    print("Generating %d in-distribution samples over %d families ..."
          % (n_id, len(ID_FAMILIES)))
    t0 = time.time()
    id_samples = generate_corpus(balanced_plan(n_id, ID_FAMILIES),
                                 seed=args.seed, progress=False)
    print("  done in %.1fs" % (time.time() - t0))

    print("Generating %d out-of-distribution samples over %d held-out families ..."
          % (n_ood, len(OOD_FAMILIES)))
    t0 = time.time()
    ood_samples = generate_corpus(balanced_plan(n_ood, OOD_FAMILIES),
                                  seed=args.seed + 1, progress=False)
    print("  done in %.1fs" % (time.time() - t0))

    df_id = pd.DataFrame([s.to_row() for s in id_samples])
    df_ood = pd.DataFrame([s.to_row() for s in ood_samples])

    # Cross-pool de-duplication (uniqueness is enforced per call, not across).
    before = len(df_ood)
    df_ood = df_ood[~df_ood["uid"].isin(set(df_id["uid"]))]
    if before != len(df_ood):
        print("  dropped %d OOD samples colliding with the ID pool"
              % (before - len(df_ood)))

    df_id = stratified_split(df_id, seed=args.seed)
    df_ood["split"] = "test_ood"
    df = pd.concat([df_id, df_ood], ignore_index=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ---------------------------------------------------------------- outputs
    full = os.path.join(args.out_dir, "sqli_corpus.parquet")
    df.to_parquet(full, index=False)
    for sp in ("train", "val", "test_id", "test_ood"):
        sub = df[df["split"] == sp]
        sub.to_csv(os.path.join(args.out_dir, "sqli_%s.csv.gz" % sp),
                   index=False, compression="gzip")

    # ---------------------------------------------------------------- summary
    stats = {
        "n_total": int(len(df)),
        "n_unique_code": int(df["uid"].nunique()),
        "n_families": int(df["family"].nunique()),
        "ood_families": OOD_FAMILIES,
        "class_balance": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "splits": {k: int(v) for k, v in df["split"].value_counts().items()},
        "split_positive_rate": {
            sp: round(float(df.loc[df["split"] == sp, "label"].mean()), 4)
            for sp in ("train", "val", "test_id", "test_ood")},
        "api_counts": {k: int(v) for k, v in df["api"].value_counts().items()},
        "difficulty_counts": {k: int(v) for k, v in df["difficulty"].value_counts().items()},
        "hard_negatives": int(df["hard_negative"].sum()),
        "lines_per_sample": {
            "mean": round(float(df["n_lines"].mean()), 2),
            "min": int(df["n_lines"].min()),
            "max": int(df["n_lines"].max()),
            "median": float(df["n_lines"].median()),
        },
        "rationale_lines_per_sample": {
            "mean": round(float(df["rationale_lines"].str.count(",").add(1).mean()), 2),
        },
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    per_fam = (df.groupby(["family", "label", "difficulty", "hard_negative"])
                 .agg(n=("uid", "size"), mean_lines=("n_lines", "mean"))
                 .reset_index()
                 .sort_values("family"))
    per_fam["mean_lines"] = per_fam["mean_lines"].round(1)
    per_fam["in_ood_split"] = per_fam["family"].isin(OOD_FAMILIES).astype(int)
    per_fam.to_csv(os.path.join(args.out_dir, "family_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("corpus written to %s" % full)
    print("=" * 70)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
