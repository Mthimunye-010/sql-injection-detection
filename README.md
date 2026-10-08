# NCOS734 Mini-Project (PG 4)

## Attention-Guided Neural Code Search for Automated SQL Injection Vulnerability Detection in Python Source Code

**AI area:** Searching / Software Engineering &nbsp;·&nbsp; **Vulnerability class:** CWE-89

Static analysers find SQL-injection flaws by matching syntactic patterns. They
are good at *finding* them and bad at *not crying wolf*: safe code that merely
looks dangerous — an integer cast, an allow-listed column name, a
regex-validated token — trips the same rules. This project builds **AGNCS**, a
neural model trained with three signals at once (detection, supervised
attention, contrastive search) that attacks that specific failure.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py --n-total 60000   # ~15 s, writes data/
jupyter notebook NCOS734_PG4_Attention_Guided_Neural_Code_Search.ipynb
```

The notebook runs top to bottom with **Run All** (≈35–50 min on a 6-thread CPU;
set `QUICK = True` in §1 for a ~6-minute smoke run). The corpus is regenerated
automatically if `data/` is missing.

To reproduce every table and figure without the notebook:

```bash
python scripts/run_experiments.py
```

To check the submission is intact (sources parse, notebook cells resolve, corpus
is valid and de-duplicated, no train/test leakage, result tables present):

```bash
python scripts/verify_submission.py
```

---

## What is in here

```
NCOS734_PG4_Attention_Guided_Neural_Code_Search.ipynb   main deliverable
src/
  vocabpools.py      lexical pools shared by vulnerable and safe generators
  sqli_corpus.py     records, function builder, SQL shape grammar
  families_vuln.py   the 15 vulnerable code families (V01-V15)
  families_safe.py   the 15 safe families (S01-S15), 8 of them hard negatives
  corpus_builder.py  assembles family bodies into valid Python, bulk generation
  code_tokenizer.py  SQLi-aware lexer, identifier anonymisation, vocabulary
  data.py            encoding to padded tensors, disk cache, subsampling
  models.py          AGNCS architecture, the three losses, ablation registry
  baselines.py       two static analysers + TF-IDF/logistic-regression
  metrics.py         detection, retrieval and attention metrics
  train.py           training loop, family-balanced sampler, early stopping
  experiment.py      orchestration shared by the script and the notebook
  viz.py             all report figures
scripts/
  generate_dataset.py   build the corpus and its splits
  run_experiments.py    the full comparative study
  ablate_trainsize.py   training-set-size ablation
  ablate_receptive_field.py  taint span vs receptive field
  build_notebook.py     regenerates the notebook from source
  verify_submission.py  pre-submission self-check
data/      corpus (parquet + per-split csv.gz), statistics, family inventory
results/   tables/ figures/ models/ raw/
```

---

## The dataset

**60,000 unique, syntactically valid Python functions**, balanced 50/50, over
**30 vulnerability/safety families**, each carrying:

| field | meaning |
|---|---|
| `code` | the function source |
| `label` | 1 = SQL-injection vulnerable (CWE-89) |
| `family` | one of 30 patterns — the relevance key for code search |
| `rationale_lines` | **statement-level taint / safety-evidence annotation** |
| `api` | sqlite3 · psycopg2 · mysql.connector · pymysql · pyodbc |
| `difficulty`, `hard_negative` | analysis metadata |

### Why synthesise rather than use Juliet / Devign / Big-Vul?

The brief names Juliet/SARD, Devign, CodeSearchNet and Big-Vul. None supplies
what this experiment needs:

| Dataset | Problem |
|---|---|
| Juliet / SARD | Mostly C/C++/Java; the Python subset misses modern ORM and f-string idioms |
| Devign, Big-Vul | C/C++ only, and function-level binary labels — **no statement-level taint annotation** |
| CodeSearchNet | Python and large, but **no vulnerability labels at all** |

The attention-guidance loss needs a token-level target, and retrieval needs a
family label. Generating the corpus from a taint-aware grammar supplies both
*exactly*, because the generator knows the flow it constructed. The cost — the
corpus cannot contain idioms its author did not think of — is the main
limitation of the work and is argued in §14 of the notebook.

### Keeping the task honest

* Vulnerable and safe samples draw identifiers, tables, docstrings, decorators
  and comments from the **same pools**; no name predicts the label.
* Comments are adversarial (`# input is already sanitised` appears on both
  classes).
* **8 hard-negative families** are safe code that still interpolates into SQL:
  integer casts, allow-listed identifiers, enum fragment lookups,
  regex-validated tokens, literal-only f-strings, composable-SQL APIs, SQL text
  that is never executed, non-SQL sinks.
* **6 families are held out of training entirely** and form the OOD test split.
* Every sample passes `ast.parse`; the corpus is de-duplicated on source SHA-1.

| Split | n | Role |
|---|---|---|
| `train` | 36,402 | 24 in-distribution families |
| `val` | 5,208 | threshold selection and early stopping |
| `test_id` | 10,392 | same 24 families |
| `test_ood` | 7,998 | **6 unseen families** |

---

## The model

```
tokens -> Embedding + position -> dilated residual CNN (d=1,2,4, RF=15)
       -> multi-head additive attention pooling -> code vector c
       -> detection head  P(vulnerable)
       -> search head     z on the unit sphere
```

```
L = L_cls  +  λ_a · L_att  +  λ_r · L_ret
```

* `L_cls` — binary cross-entropy on the detection head.
* `L_att` — **attention guidance**: cross-entropy between the attention map and
  the normalised statement-level annotation. This is the component the project
  title names.
* `L_ret` — supervised contrastive loss with the **vulnerability family** as the
  label, which makes the embedding a code-search index rather than a by-product
  of classification.

A dilated CNN is used instead of a Transformer encoder because, measured on the
target CPU, the Transformer runs at 96 samples/s against 376 at equal width — a
3.9× gap that decides whether a three-seed, seven-method study is feasible. The
global token-to-token interaction self-attention would add is supplied by the
attention *pooling* layer, which is where this project needs attention anyway.

The receptive field was chosen the hard way — see **Findings worth knowing**
below.

### Methods compared

| | Method | Isolates |
|---|---|---|
| M1 | Static analyser — pattern rules (semgrep / Bandit B608 class) | what CI tooling actually does |
| M2 | Static analyser — AST taint analysis | a genuinely strong symbolic baseline |
| M3 | TF-IDF + logistic regression (same token stream) | the value of the architecture |
| M4 | CNN + mean pooling | attention at all |
| M5 | CNN + unguided attention | *guiding* the attention |
| M6 | CNN + attention + contrastive head | the search head |
| **M7** | **AGNCS — guided attention + contrastive** | the full method |

---

## Findings worth knowing

Two results shape how every other number in this project should be read.

### The receptive-field hypothesis was wrong

Measuring the token distance from the first to the last annotated statement
gives a median span of **37 tokens in-distribution and 58 in the held-out
families**. That predicts the encoder's receptive field must be wide enough to
span the taint path, or it cannot represent "tainted value here *and* a sink
there".

`scripts/ablate_receptive_field.py` tested R in {15, 31, 63, 127} and the
prediction failed. The *smallest* field was at least as good everywhere, and a
wider one correlated with a **worse** held-out false-positive rate
(Spearman ρ = +0.80) and worse held-out ROC-AUC (ρ = -0.80), non-monotonically.

The likely reason: the decisive evidence is **local**. The quoted-interpolation
motif next to SQL keywords fits inside 15 tokens, so a wider field adds no
necessary information and gives the model room to fit family-specific long-range
layout — which is exactly what fails to transfer.

### In-distribution validation cannot select for OOD generalisation

Validation F1 is ~0.999 for *every* configuration, so early stopping picks an
essentially arbitrary epoch (`best_epoch` swings 5, 4, 2, 4 across the ablation)
and the held-out score is dominated by seed noise: held-out F1 varies by 0.121
across configurations against 0.0013 in-distribution, a ratio of about 90.

Applied throughout:

* The receptive field is selected **on validation** (where all four
  configurations tie) with cost as the tie-breaker, giving dilations 1,2,4.
  Choosing it by the held-out score would be selecting on the test set.
* Every held-out number is reported as mean ± sd over three seeds. Check that
  width before believing any single OOD comparison.
* The single-seed receptive-field ablation is reported as **inconclusive about
  which field is better** — it is evidence that this axis cannot be tuned with
  the signal available, not that small is better.

---

## Metrics

**Detection** — Accuracy, Precision, Recall, F1, **FPR**, Specificity, Balanced
Accuracy, MCC, ROC-AUC, PR-AUC. FPR is the headline: the project's premise is
that static analysers are too noisy to use, not that they miss too much.
ROC/PR-AUC are undefined for the rule-based analysers (they emit a hard
decision) and are reported as `--`.

**Code search** — MRR, Recall@{1,5,10}, P@k, MAP, nDCG@10, with *same
vulnerability family* as the relevance criterion; plus a transfer-search task
where OOD vulnerable queries are run against an in-distribution corpus.

**Attention quality** — attention mass on the annotation, **lift over uniform**
(1.0 = no better than looking everywhere), pointing-game accuracy, and top-k IoU.

**Protocol** — 3 seeds per neural variant, mean ± sd, paired *t*-test and
Wilcoxon signed-rank with Cohen's *d_z*. Thresholds and early stopping use `val`
only; the test splits are read once.

---

## Reproducibility

Fixed seeds throughout (corpus `20260404`; training `0,1,2`). The vocabulary is
built on the training split alone. The encoded corpus is cached under
`data/cache/` with a key covering the tokeniser configuration, so changing the
lexer invalidates it automatically. `results/raw/environment.json` records the
exact library versions and hardware of the run.
