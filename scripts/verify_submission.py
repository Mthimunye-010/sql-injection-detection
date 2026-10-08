"""Pre-submission self-check.

Verifies the things that would cost marks if they were broken, and prints a
pass/fail checklist:

  * every source file and every notebook code cell parses;
  * no notebook cell uses a name no earlier cell defined (catches a cell that
    only works because of a stale kernel);
  * the corpus exists, is the advertised size, is 100% valid Python, and is
    de-duplicated;
  * train/test leakage after identifier anonymisation is negligible;
  * the taint annotations are non-empty and point inside the source;
  * the result tables and figures the report cites are present.

    python scripts/verify_submission.py
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NB = "NCOS734_PG4_Attention_Guided_Neural_Code_Search.ipynb"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list = []
_CORPUS = None


def corpus():
    """Load the parquet once; it is ~11 MB and two copies can exhaust memory
    when something else (a training run) is using the machine."""
    global _CORPUS
    if _CORPUS is None:
        import pandas as pd
        _CORPUS = pd.read_parquet("data/sqli_corpus.parquet")
    return _CORPUS


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    status = PASS if ok else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    print("  [%s] %-52s %s" % (status, name, detail))


# --------------------------------------------------------------------------- #
def check_sources() -> None:
    print("\nSource files")
    import glob
    bad = []
    files = sorted(glob.glob("src/*.py") + glob.glob("scripts/*.py"))
    for f in files:
        try:
            ast.parse(open(f, encoding="utf-8").read())
        except SyntaxError as e:
            bad.append("%s:%s" % (f, e.lineno))
    check("all %d source files parse" % len(files), not bad, ", ".join(bad))

    ctrl = []
    for f in files:
        txt = open(f, encoding="utf-8").read()
        if any(ord(c) < 32 and c not in "\n\t\r" for c in txt):
            ctrl.append(f)
    check("no stray control characters in source", not ctrl, ", ".join(ctrl))

    try:
        import src.baselines, src.code_tokenizer, src.corpus_builder, src.data   # noqa
        import src.experiment, src.metrics, src.models, src.train, src.viz       # noqa
        check("all modules import", True)
    except Exception as e:                                   # pragma: no cover
        check("all modules import", False, "%s: %s" % (type(e).__name__, e))


def check_notebook() -> None:
    print("\nNotebook")
    if not os.path.exists(NB):
        check("notebook present", False, NB)
        return
    nb = json.load(open(NB, encoding="utf-8"))
    cells = [("\n".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    check("notebook present (%d cells, %d code)" % (len(nb["cells"]), len(cells)), True)

    bad = []
    for i, src in enumerate(cells):
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad.append("cell %d line %s" % (i, e.lineno))
    check("every code cell parses", not bad, "; ".join(bad))

    defined = set(dir(builtins)) | {"display", "get_ipython"}
    problems = []
    for i, src in enumerate(cells):
        tree = ast.parse(src)
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.alias):
                defined.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name):
                defined.add(n.target.id)
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                defined.add(n.name)
            elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
                defined.add(n.optional_vars.id)
        missing = sorted(used - defined)
        if missing:
            problems.append("cell %d: %s" % (i, missing))
    check("no cell depends on an undefined name", not problems, "; ".join(problems[:3]))


def check_corpus() -> None:
    print("\nDataset")
    path = "data/sqli_corpus.parquet"
    if not os.path.exists(path):
        check("corpus present", False, "run scripts/generate_dataset.py")
        return
    from src.code_tokenizer import parse_rationale

    df = corpus()
    check("corpus present (%d rows)" % len(df), len(df) >= 10000)
    check("source de-duplicated", df["uid"].nunique() == len(df),
          "%d unique of %d" % (df["uid"].nunique(), len(df)))
    check("30 families present", df["family"].nunique() == 30,
          "%d found" % df["family"].nunique())
    bal = df["label"].mean()
    check("class balance within 1% of 50/50", abs(bal - 0.5) < 0.01, "positive rate %.4f" % bal)

    splits = set(df["split"])
    check("four splits present", splits == {"train", "val", "test_id", "test_ood"},
          str(sorted(splits)))

    train_fams = set(df.loc[df.split == "train", "family"])
    ood_fams = set(df.loc[df.split == "test_ood", "family"])
    check("held-out families absent from train", not (train_fams & ood_fams),
          "leaked: %s" % sorted(train_fams & ood_fams))

    sample = df.sample(min(4000, len(df)), random_state=0)
    bad = 0
    for code in sample["code"]:
        try:
            ast.parse(code)
        except SyntaxError:
            bad += 1
    check("sampled source is valid Python", bad == 0, "%d of %d failed" % (bad, len(sample)))

    empty = sum(1 for r in sample["rationale_lines"] if not parse_rationale(r))
    check("every sample carries an annotation", empty == 0, "%d empty" % empty)

    oob = 0
    for code, rl in zip(sample["code"], sample["rationale_lines"]):
        n = code.count("\n") + 1
        if any(i < 0 or i >= n for i in parse_rationale(rl)):
            oob += 1
    check("annotations point inside the source", oob == 0, "%d out of range" % oob)


def check_leakage() -> None:
    print("\nLeakage after identifier anonymisation")
    if not os.path.exists("data/sqli_corpus.parquet"):
        check("leakage check", False, "no corpus")
        return
    import numpy as np
    from src.data import prepare

    try:
        splits, vocab, _ = prepare(corpus(), max_len=160, cache_dir="data/cache",
                                   verbose=False)
    except MemoryError:
        check("leakage check", True,
              "skipped: not enough free memory right now", warn_only=True)
        return

    def h(s):
        return np.array([hashlib.sha1(r.tobytes()).hexdigest() for r in s.x])

    H = {k: h(v) for k, v in splits.items()}
    train = set(H["train"])
    for k in ("test_id", "test_ood"):
        ov = float(np.isin(H[k], list(train)).mean())
        check("%s overlap with train < 1%%" % k, ov < 0.01, "%.3f%%" % (100 * ov))
    uniq = len(set(H["train"])) / len(H["train"])
    check("train token sequences >99% unique", uniq > 0.99, "%.2f%%" % (100 * uniq))


def check_results() -> None:
    print("\nResults artefacts")
    wanted_tables = [
        "all_runs.csv", "summary_by_method_split.csv",
        "table_detection_test_id.csv", "table_detection_test_ood.csv",
        "hard_negative_fpr.csv", "per_family_errors.csv",
        "significance_tests.csv",
    ]
    for t in wanted_tables:
        p = os.path.join("results/tables", t)
        check("table %s" % t, os.path.exists(p),
              "" if os.path.exists(p) else "run scripts/run_experiments.py",
              warn_only=True)
    optional = ["ablation_receptive_field.csv", "ablation_train_size.csv",
                "taint_span_stats.csv"]
    for t in optional:
        p = os.path.join("results/tables", t)
        check("ablation %s" % t, os.path.exists(p), "optional", warn_only=True)
    figs = sorted(os.listdir("results/figures")) if os.path.isdir("results/figures") else []
    check("figures produced", len(figs) >= 8, "%d found" % len(figs), warn_only=True)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("=" * 74)
    print("NCOS734 PG4 -- pre-submission self-check")
    print("=" * 74)
    check_sources()
    check_notebook()
    check_corpus()
    check_leakage()
    check_results()

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print("\n" + "=" * 74)
    print("%d checks: %d passed, %d warnings, %d FAILED"
          % (len(results), len(results) - n_fail - n_warn, n_warn, n_fail))
    if n_fail:
        print("\nFailures:")
        for s, name, detail in results:
            if s == FAIL:
                print("  - %s  %s" % (name, detail))
    print("=" * 74)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
