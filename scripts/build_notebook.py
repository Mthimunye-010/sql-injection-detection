"""Build the deliverable Jupyter notebook.

Keeping the notebook in a generator script means the narrative, the maths and
the code stay in version-controllable text instead of a 1-line JSON blob.
Run:  python scripts/build_notebook.py
"""

from __future__ import annotations

import json
import os
import sys

NB = "NCOS734_PG4_Attention_Guided_Neural_Code_Search.ipynb"

cells = []


def md(text: str) -> None:
    lines = text.strip("\n").splitlines(keepends=True)
    lines = [line for line in lines if line.strip() != "---"]
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": lines})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# =========================================================================== #
md(r"""
# NCOS734 Mini-Project (PG 4)

# Attention-Guided Neural Code Search for Automated SQL Injection Vulnerability Detection in Python Source Code

**AI area:** Searching / Software Engineering &nbsp;&nbsp;|&nbsp;&nbsp; **Vulnerability class:** CWE-89

---

### What this notebook does

Static analysers find SQL-injection flaws by matching syntactic patterns. They
are good at *finding* them and bad at *not crying wolf*: safe code that merely
looks dangerous — an integer cast, an allow-listed column name, a
regex-validated token — trips the same rules. The alerts an analyst discards
are the cost of the tool.

This notebook builds **AGNCS**, a neural model that attacks that specific
failure. It is trained with three signals at once:

1. a **detection** head (is this function vulnerable?),
2. an **attention-guidance** loss that supervises *where the model looks*,
   using statement-level taint annotations, and
3. a **contrastive search** head that turns the encoder into a code-search
   index, so the model can be queried by example.

It runs top to bottom with *Run All*. Everything it needs is in this project
folder.

### Contents

| § | Section |
|---|---|
| 1 | Environment and configuration |
| 2 | Problem formulation |
| 3 | The dataset: a taint-annotated synthetic corpus |
| 4 | Tokenisation: making interpolation visible |
| 5 | Algorithms and pseudocode |
| 6 | Evaluation metrics |
| 7 | Baselines: two static analysers and a classical classifier |
| 8 | Training the neural variants |
| 9 | Results |
| 10 | Neural code search in action |
| 11 | Attention case study |
| 12 | Sanity check on hand-written code |
| 13 | Statistical significance |
| 14 | Limitations and conclusions |
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 1. Environment and configuration
""")

code(r'''
import os, sys, json, time, platform, random, warnings
warnings.filterwarnings("ignore")

from IPython.display import HTML, display
display(HTML("""<style>
.jp-RenderedMarkdown, .text_cell_render { font-family: 'Aptos', 'Segoe UI', sans-serif; line-height: 1.55; }
.jp-RenderedMarkdown h1, .jp-RenderedMarkdown h2, .text_cell_render h1, .text_cell_render h2 { font-weight: 650; letter-spacing: -0.02em; }
pre, code, .jp-InputArea-editor, .CodeMirror { font-family: 'Cascadia Code', 'Consolas', monospace; }
</style>"""))

PROJECT_ROOT = os.path.abspath(".")
if not os.path.isdir(os.path.join(PROJECT_ROOT, "src")):
    PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

from src import viz
from src.data import prepare, subsample
from src.experiment import (ATT_METRICS, MAIN_METRICS, METHOD_LABELS,
                            METHOD_ORDER, SEARCH_METRICS,
                            eval_cross_pattern_retrieval, hard_negative_fpr,
                            per_family_table_multiseed, pretty_table,
                            run_neural, run_static_baselines,
                            run_tfidf_baseline, significance_table,
                            summarise, tfidf_retrieval)
from src.models import AGNCS, VARIANTS, VARIANT_LABELS, receptive_field

GLOBAL_SEED = 20260404
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
torch.set_num_threads(6)
viz.set_style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

print("python      ", sys.version.split()[0])
print("platform    ", platform.platform())
print("processor   ", platform.processor())
print("torch       ", torch.__version__, "| threads:", torch.get_num_threads())
print("numpy       ", np.__version__, "| pandas", pd.__version__)
print("project root", PROJECT_ROOT)
''')

md(r"""
### Experiment configuration

`N_TRAIN` is the training-set size. §9.5 shows the ablation that justifies it:
performance saturates well below the full 36,402 training samples, so a subset
buys a tractable notebook runtime at no measurable cost in accuracy.

Set `QUICK = True` for a ~6-minute smoke run (1 seed, small subsets). The
default configuration is the one the report's numbers come from and takes
roughly **35–50 minutes** on a 6-thread CPU.
""")

code(r'''
QUICK = False          # True -> fast smoke run; False -> the reported configuration

MAX_LEN   = 160        # tokens per function (head+tail truncation beyond this)
D_MODEL   = 96         # encoder width
DILATIONS = (1, 2, 4)  # receptive field 15 tokens -- chosen in 4.2 / 9.5
EPOCHS    = 5
N_TRAIN   = 12000      # stratified subset of the 36,402 training samples
N_VAL     = 4000
SEEDS     = [0, 1, 2]
BATCH     = 128

if QUICK:
    SEEDS, N_TRAIN, N_VAL, EPOCHS = [0], 4000, 2000, 3

CFG = dict(epochs=EPOCHS, batch_size=BATCH, d_model=D_MODEL, patience=2,
           lr=2.5e-3, families_per_batch=16, dilations=DILATIONS)
print(json.dumps({"QUICK": QUICK, "MAX_LEN": MAX_LEN, "N_TRAIN": N_TRAIN,
                  "SEEDS": SEEDS, **CFG}, indent=2, default=str))
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 2. Problem formulation

### 2.1 The two coupled tasks

Let $x$ be the token sequence of a Python function, $x = (x_1,\dots,x_L)$,
$x_t \in \mathcal{V}$.

**Task 1 — detection.** Learn $f_\theta : \mathcal{V}^L \to [0,1]$ estimating
$P(y=1 \mid x)$, where $y=1$ means the function contains a SQL-injection flaw
(CWE-89). The decision is $\hat{y} = \mathbb{1}[f_\theta(x) \ge \tau]$ with the
threshold $\tau$ chosen on validation data.

**Task 2 — code search.** Learn $g_\theta : \mathcal{V}^L \to \mathbb{S}^{d-1}$
mapping a function onto the unit sphere so that cosine similarity ranks
semantically related code first. With a query $q$ and corpus $\mathcal{C}$, the
retrieved list is

$$\text{rank}(q) = \operatorname*{arg\,sort}_{c \in \mathcal{C}}\; -\,g_\theta(q)^{\top} g_\theta(c).$$

The two tasks share one encoder. That sharing is the point: the representation
that decides *vulnerable or not* is the same one that decides *similar or not*.

### 2.2 What makes a function vulnerable

A SQL injection exists when a **taint path** connects an untrusted source to a
query-construction sink without passing a sanitiser. Writing $S$ for sources,
$K$ for sinks and $N$ for sanitisers, the function is vulnerable iff there is a
path in the data-flow graph $G=(V,E)$

$$\exists\, v_0 \in S,\; v_n \in K : \quad v_0 \to v_1 \to \cdots \to v_n \in E^{*}
\quad\text{and}\quad \{v_0,\dots,v_n\} \cap N = \emptyset .$$

The tell-tale detail is *how* the value reaches the sink. Compare

```python
cur.execute(f"SELECT * FROM t WHERE u = '{name}'")   # interpolated  -> vulnerable
cur.execute("SELECT * FROM t WHERE u = ?", (name,))  # bound         -> safe
```

Lexically these are nearly identical. Semantically one concatenates attacker
text into the statement and the other hands it to the driver as data. **Every
useful signal in this problem is a structural relation, not a word.**

### 2.3 Why this is a *searching* problem

Deciding membership of the set above is a reachability search over the data-flow
graph. Classical static analysis performs that search symbolically, which is
exact for the patterns it encodes and blind to everything else. This project
replaces the symbolic search with a *learned* one: the encoder embeds code so
that functions sharing a taint pattern land near one another, and detection
becomes a question about a neighbourhood in that space. Attention is the
mechanism that decides which tokens the search attends to — and the central
hypothesis is that **supervising the attention makes the learned search
generalise to patterns never seen in training.**

### 2.4 Variables

| Symbol | Meaning |
|---|---|
| $x \in \mathcal{V}^L$ | token sequence of one function; $L$ = `MAX_LEN` |
| $y \in \{0,1\}$ | 1 = vulnerable (CWE-89) |
| $\phi \in \{1..30\}$ | vulnerability-family label (the retrieval relevance key) |
| $m \in \{0,1\}^L$ | taint / safety-evidence annotation per token |
| $H \in \mathbb{R}^{L \times d}$ | contextual token states from the encoder |
| $\alpha \in \Delta^{L-1}$ | attention distribution over tokens |
| $c \in \mathbb{R}^{d}$ | pooled code vector |
| $z \in \mathbb{S}^{d'-1}$ | L2-normalised search embedding |
| $\lambda_a, \lambda_r$ | weights of the guidance and contrastive losses |
| $\tau$ | decision threshold, selected on validation |
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 3. The dataset: a taint-annotated synthetic corpus

### 3.1 Why not Juliet / SARD / Devign / Big-Vul directly?

The brief lists Juliet Test Suite, SARD, Devign, CodeSearchNet and Big-Vul. They
do not fit this experiment, for concrete reasons:

| Dataset | Problem for *this* study |
|---|---|
| Juliet / SARD | Overwhelmingly C/C++/Java; the Python subset is small and does not cover modern ORM and f-string idioms |
| Devign, Big-Vul | C/C++ only, and labels are function-level binary — **no statement-level taint annotation**, which the guidance loss requires |
| CodeSearchNet | Python, large, but has **no vulnerability labels at all** — it is a docstring-retrieval corpus |

This project needs all three of: a binary label, a **statement-level taint
annotation**, and a **family label** for retrieval relevance. No public corpus
supplies the combination for Python, so we synthesise it from a taint-aware
grammar. The annotation is then exact by construction rather than hand-labelled,
which removes the label noise that mined vulnerability datasets are known for.

The honest cost of this choice is discussed in §14 and is the main limitation of
the work.

### 3.2 Corpus design

**30 families, 60,000 unique functions**, balanced 50/50:

* **15 vulnerable families** (V01–V15) spanning direct f-string interpolation,
  `%`-formatting, `str.format`, concatenation, multi-hop aliasing,
  inter-procedural helpers, request-dictionary sources, loop-built `WHERE`
  clauses, ORM raw-SQL escape hatches, `executescript`, `string.Template`,
  identifier injection in `ORDER BY`, fragment joining, partial
  parameterisation, and ineffective sanitisers.
* **15 safe families** (S01–S15). Eight are **hard negatives**: safe code that
  still interpolates into a SQL string (integer casts, allow-listed identifiers,
  enum fragment lookups, regex-validated tokens, literal-only f-strings,
  composable-SQL APIs, SQL text that is never executed, non-SQL sinks). These
  are the patterns that make static analysers fire wrongly.

Controls that keep the task honest:

* Vulnerable and safe samples draw identifiers, tables, columns, docstrings,
  decorators and comments from the **same pools**, so no name predicts the label.
* Comments are adversarial: `# input is already sanitised` appears on both
  classes.
* Five database APIs, seven wrapping styles (plain function, `try/except`,
  `with`, class method, …), optional type hints and 0–3 distractor statements
  per sample are all sampled **independently of the label**.
* Every sample is validated with `ast.parse`, and the corpus is de-duplicated on
  the SHA-1 of the source.

### 3.3 Splits

| Split | n | Role |
|---|---|---|
| `train` | 36,402 | 24 in-distribution families, 70% |
| `val` | 5,208 | same 24 families, 10% — threshold and early stopping |
| `test_id` | 10,392 | same 24 families, 20% |
| `test_ood` | 7,998 | **6 families held out entirely from training** |

The OOD split is the experiment that matters. Anyone can fit a grammar they were
trained on; the question is whether the learned notion of a taint flow transfers
to injection patterns the model has never seen.

Two properties of that split are deliberate and should be read before the
results:

* The six held-out families are all rated `hard` — inter-procedural helpers,
  `string.Template` substitution and fragment joining on the vulnerable side;
  enum fragment lookups, never-executed SQL text and composable-SQL APIs on the
  safe side.
* **All three held-out safe families are hard negatives.** The OOD split
  therefore contains no "easy" safe code at all, which makes its false-positive
  rate a pure hard-negative measure and the hardest number in this report. It is
  not comparable to the in-distribution FPR, which averages over easy and hard
  negatives alike — §9.2 separates the two so the comparison is fair.
""")

code(r'''
CORPUS = "data/sqli_corpus.parquet"
if not os.path.exists(CORPUS):
    print("corpus not found -- generating (about 15 s) ...")
    os.system(f'"{sys.executable}" scripts/generate_dataset.py --n-total 60000 --out-dir data')

df = pd.read_parquet(CORPUS)
stats = json.load(open("data/corpus_stats.json", encoding="utf-8"))

print("samples              {:,}  (unique source: {:,})".format(stats["n_total"], stats["n_unique_code"]))
print("families             %d   (%d held out for the OOD split)" % (stats["n_families"], len(stats["ood_families"])))
print("class balance        safe {:,} / vulnerable {:,}".format(stats["class_balance"]["0"], stats["class_balance"]["1"]))
print("hard negatives       {:,}".format(stats["hard_negatives"]))
print("lines per function   mean %.1f, median %.0f, range %d-%d" % (
    stats["lines_per_sample"]["mean"], stats["lines_per_sample"]["median"],
    stats["lines_per_sample"]["min"], stats["lines_per_sample"]["max"]))
print("annotated statements mean %.2f per function" % stats["rationale_lines_per_sample"]["mean"])
print("\nsplits:          ", stats["splits"])
print("positive rate:   ", stats["split_positive_rate"])
print("held-out families:", ", ".join(stats["ood_families"]))
''')

code(r'''
# Family inventory -- the table that goes in the report's dataset section.
fam = pd.read_csv("data/family_summary.csv").sort_values(["label", "family"], ascending=[False, True])
fam["kind"] = np.where(fam.label == 1, "vulnerable", "safe")
fam["hard_negative"] = fam["hard_negative"].map({1: "yes", 0: ""})
fam["in_ood_split"] = fam["in_ood_split"].map({1: "held out", 0: ""})
display(fam[["family", "kind", "difficulty", "hard_negative", "in_ood_split", "n", "mean_lines"]]
        .rename(columns={"n": "samples", "mean_lines": "avg lines"})
        .reset_index(drop=True))
''')

md(r"""
### 3.4 What the data actually looks like

The contrast below is the whole problem in miniature. Lines marked `>>` are the
statement-level annotation the guidance loss uses: for a vulnerable sample these
are the taint path, for a safe sample the **safety evidence** (the binding, the
cast, the validation guard). Annotating both classes means the guidance signal
is available for every sample, not just the positives.
""")

code(r'''
from src.code_tokenizer import parse_rationale

def show(family, n=1, width=96):
    sub = df[df.family == family].head(n)
    for _, r in sub.iterrows():
        rl = set(parse_rationale(r.rationale_lines))
        tag = "VULNERABLE (CWE-89)" if r.label else "SAFE"
        hard = "  [HARD NEGATIVE]" if r.hard_negative else ""
        print("=" * width)
        print("%s   %s%s   api=%s   difficulty=%s" % (r.family, tag, hard, r.api, r.difficulty))
        print("-" * width)
        for i, ln in enumerate(r.code.split("\n")):
            print("%s %2d | %s" % (">>" if i in rl else "  ", i, ln))
        print()

show("V01_fstring_direct")
show("S01_qmark_params")
''')

code(r'''
# The hard negatives: safe code that a pattern-matching analyser cannot clear.
for f in ["S06_int_cast", "S07_allowlisted_identifier", "S09_regex_validated"]:
    show(f)
''')

code(r'''
# And the harder vulnerabilities: taint that travels before it reaches the sink.
for f in ["V05_multi_hop_alias", "V12_order_by_identifier"]:
    show(f)
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 4. Tokenisation: making interpolation visible

A model that sees a SQL string as an opaque blob cannot possibly separate
`'{name}'` from `?`. The lexer in `src/code_tokenizer.py` therefore makes three
decisions that carry most of the inductive bias.

**1. Per-sample identifier indexing.** Any identifier outside a fixed API
lexicon becomes `<ID_k>`, where $k$ is its order of first appearance *in that
sample*. This destroys name memorisation — `user_input` is no more suspicious
than `tenant` — while preserving **aliasing**: a variable written on line 4 and
read on line 7 is `<ID_3>` in both places, so the data flow survives. That is
precisely the signal a taint analysis needs.

**2. String-literal decomposition.** Each literal is expanded into structural
markers:

| Source | Token stream | Meaning |
|---|---|---|
| `f"... u = '{v}'"` | `<SQ> <FFIELD_BEG> <ID_k> <FFIELD_END> <SQ>` | value spliced into a quoted SQL slot |
| `"... u = ?"` | `<QMARK>` | driver placeholder |
| `"... u = %s"` | `<PCT_FMT>` | driver placeholder |
| `"... u = :v"` | `<COLON_PARAM>` | named bind |

**3. A hand-rolled scanner rather than `tokenize`.** CPython 3.12 changed
f-string tokenisation (PEP 701), so the standard library would give different
token streams on different interpreters. This notebook has to reproduce on the
marker's machine.

The resulting vocabulary is small (≈400 types) because almost everything is
normalised — which is the goal: the model is forced to learn structure.
""")

code(r'''
from src.code_tokenizer import tokenise, rationale_mask

ex_v = df[df.family == "V01_fstring_direct"].iloc[0]
ex_s = df[df.family == "S01_qmark_params"].iloc[0]

for ex, name in [(ex_v, "VULNERABLE -- interpolated"), (ex_s, "SAFE -- parameterised")]:
    toks, lines = tokenise(ex.code)
    mask = rationale_mask(lines, parse_rationale(ex.rationale_lines))
    print("=" * 96); print(name); print("-" * 96)
    print("tokens on the annotated statements (%d of %d):" % (sum(mask), len(mask)))
    print(" ".join(t for t, m in zip(toks, mask) if m)); print()
''')

code(r'''
# Encode every split.  The vocabulary is built on `train` only -- building it on
# the full corpus would leak held-out token statistics into the model.
t0 = time.time()
splits, vocab, frames = prepare(df, max_len=MAX_LEN, cache_dir="data/cache")
print("\nprepared in %.1fs | vocabulary = %d token types" % (time.time() - t0, len(vocab)))

lens = np.array([s.valid.sum(1).mean() for s in splits.values()])
for k, v in splits.items():
    print("  %-9s n=%6d   mean length %5.1f tokens   annotation density %.3f"
          % (k, len(v), v.valid.sum(1).mean(), v.mask.sum() / v.valid.sum()))
''')

md(r"""
### 4.1 Integrity check: does anonymisation create leakage?

Identifier anonymisation is the right thing to do, but it carries a risk worth
checking rather than assuming away: if two distinct functions differ only in a
variable name, they collapse to the **same token sequence**, and a test sample
could be an exact copy of a training sample in the only representation the model
ever sees. The cell below measures that directly on the encoded tensors.
""")

code(r'''
import hashlib
def seq_hashes(s):
    return np.array([hashlib.sha1(r.tobytes()).hexdigest() for r in s.x])

H = {k: seq_hashes(v) for k, v in splits.items()}
for k, h in H.items():
    print("%-9s n=%6d   unique token sequences %6d  (%.2f%%)"
          % (k, len(h), len(set(h)), 100 * len(set(h)) / len(h)))
train_set = set(H["train"])
print()
for k in ("val", "test_id", "test_ood"):
    ov = np.isin(H[k], list(train_set))
    print("%-9s identical to a TRAIN sequence: %4d / %5d = %.2f%%"
          % (k, ov.sum(), len(ov), 100 * ov.mean()))
print("\nLow overlap means the combinatorial diversity of the generator "
      "(tables, columns, SQL shapes, wrapping styles, distractors) survives "
      "anonymisation, so the test splits are not measuring memorisation.")
''')

md(r"""
### 4.2 How wide is a taint path?

A dilated CNN output position sees $R = 1 + (k-1)\sum_i d_i$ tokens. If $R$ is
smaller than the distance between a taint source and its sink, the encoder
cannot represent the conjunction "tainted value here *and* a SQL sink there",
no matter how much data it gets. So the obvious first step is to measure the
span.

The measurement below says the span is wide -- median 37 tokens
in-distribution, 58 in the held-out families -- which predicts that a wide
receptive field should be necessary. **§9.5 tests that prediction and refutes
it.** Read the two together: the negative result is the more useful one, and it
is why `DILATIONS` is set to the *smallest* candidate rather than the widest.

""")

code(r'''
CANDIDATE_DILATIONS = [(1, 2, 4), (1, 2, 4, 8), (1, 2, 4, 8, 16), (1, 2, 4, 8, 16, 32)]

rows = []
for sp in ("train", "test_id", "test_ood"):
    s = splits[sp]
    spans = np.array([(lambda ix: ix[-1] - ix[0] + 1 if len(ix) else 0)(np.flatnonzero(s.mask[i]))
                      for i in range(len(s))])
    row = {"split": sp, "mean": round(spans.mean(), 1), "median": int(np.median(spans)),
           "p90": int(np.percentile(spans, 90)), "max": int(spans.max())}
    for dil in CANDIDATE_DILATIONS:
        row["RF=%d" % receptive_field(dil)] = "%.0f%%" % (100 * (spans <= receptive_field(dil)).mean())
    rows.append(row)
span_tbl = pd.DataFrame(rows)
os.makedirs("results/tables", exist_ok=True)
span_tbl.to_csv("results/tables/taint_span_stats.csv", index=False)
print("Taint-span width in tokens, and the fraction of spans each receptive field covers:\n")
display(span_tbl)
print("Chosen: dilations %s -> receptive field %d tokens."
      % (DILATIONS, receptive_field(DILATIONS)))
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 5. Algorithms and pseudocode

### 5.1 Architecture

```
                         token ids  x  (B, L)
                                 |
                    Embedding + learned position            (B, L, d)
                                 |
               Dilated residual CNN encoder, d=1,2,4    (B, L, d)
                      receptive field 15 tokens
                                 |
                +----------------+-----------------+
                |                                  |
     Multi-head additive attention pooling   -->  alpha  (B, L)
                |                                  |
                c  (B, d)                   supervised by L_att
         +------+------+
         |             |
   Detection head   Search head
   P(vulnerable)    z on the unit sphere
```

**Why a dilated CNN and not a Transformer encoder?** Measured on the target CPU
(§1), a 2-layer Transformer encoder runs at 96 samples/s against 376 for this
encoder at equal width — a 3.9× difference that decides whether a
three-seed, seven-method study is feasible at all. The global token-to-token
interaction self-attention would add is supplied instead by the attention
*pooling* layer, which is where this project needs attention anyway, because it
is the quantity being supervised.

**On the receptive field.** §4.2 measures the taint span at a median of 37
tokens in-distribution and 58 in the held-out families, which predicts that the
field should be widened to cover it. §9.5 tests $R \in \{15, 31, 63, 127\}$ and
the prediction fails: the *smallest* field is at least as good everywhere, and a
wider one correlates with a *worse* held-out false-positive rate. The dilations
are therefore $1,2,4$ ($R = 15$), selected on validation with cost as the
tie-breaker. §9.5 explains why, and why the choice could not legitimately be
made on the held-out split.

### 5.2 The attention pooling layer

$$e_{h,t} = v_h^{\top}\tanh(W_h s_t + b_h), \qquad
\alpha_{h,t} = \frac{\exp(e_{h,t})}{\sum_{t'} \exp(e_{h,t'})}\,\mathbb{1}[t \notin \text{PAD}]$$

$$c_h = \sum_t \alpha_{h,t}\, s_t, \qquad c = W_o\,[c_1; \dots; c_H], \qquad
\alpha_t = \tfrac{1}{H}\sum_h \alpha_{h,t}$$

The head-averaged $\alpha_t$ is both the explanation shown to a user and the
quantity the guidance loss acts on.

### 5.3 The three losses

**Detection** — binary cross-entropy:
$$\mathcal{L}_{cls} = -\frac{1}{B}\sum_i \Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]$$

**Attention guidance** — cross-entropy between the attention map and the
normalised annotation $q_t = m_t / \sum_{t'} m_{t'}$:
$$\mathcal{L}_{att} = -\frac{1}{|B^{+}|}\sum_{i \in B^{+}} \sum_t q_{i,t}\,\log \alpha_{i,t}$$
Minimising it maximises the attention mass landing on the annotated statements.
Samples with an empty annotation contribute nothing.

**Reading this loss correctly.** The cross-entropy has an *irreducible
floor*: it cannot fall below the target's own entropy
$H(q) = \log|m|$, which is about 3.4 nats for a typical 30-token
annotation, since $\mathcal{L}_{att} = H(q) + D_{KL}(q \,\|\, \alpha)$.
A raw $\mathcal{L}_{att}$ of 3.6 is therefore *not* "a large loss" --- it means
the attention distribution already closely matches the target. To avoid that
misreading, the training log and Figure 3b report
$$D_{KL}(q \,\|\, \alpha) = \mathcal{L}_{att} - H(q),$$
which is $0$ at a perfect match. The two differ by a parameter-independent
constant, so the gradients are identical.

**Contrastive search** — supervised contrastive loss (Khosla et al., 2020) with
the **vulnerability family** as the label:
$$\mathcal{L}_{ret} = \sum_{i}\frac{-1}{|P(i)|}\sum_{p \in P(i)}
\log\frac{\exp(z_i^{\top} z_p/\kappa)}{\sum_{a \neq i}\exp(z_i^{\top} z_a/\kappa)}$$
where $P(i)$ are the in-batch samples sharing $i$'s family and $\kappa=0.1$.
Using the *family* rather than the binary label matters: two different injection
patterns should not collapse onto the same point merely because both are
vulnerable.

$$\boxed{\;\mathcal{L} = \mathcal{L}_{cls} + \lambda_a\,\mathcal{L}_{att} + \lambda_r\,\mathcal{L}_{ret}\;}$$

### 5.4 Pseudocode

```
ALGORITHM 1  AGNCS — training
────────────────────────────────────────────────────────────────────────────
input :  D = {(x_i, y_i, m_i, phi_i)}        annotated training corpus
         lambda_a, lambda_r                  loss weights
         E, B, eta                           epochs, batch size, base LR
output:  theta*                              parameters of the best epoch

 1  theta <- init()
 2  best  <- -inf ;  stale <- 0
 3  for epoch = 1 .. E do
 4      for each batch Bt ~ FAMILY-BALANCED-SAMPLER(D, B) do
 5          eta_t <- WARMUP-COSINE(step, eta)
 6          H, padmask <- ENCODER(x)                       // (B, L, d)
 7          alpha      <- SOFTMAX(SCORE(H) masked by padmask)
 8          c          <- sum_t alpha_t * H_t
 9          p          <- sigmoid(CLS-HEAD(c))
10          z          <- NORMALISE(SEARCH-HEAD(c))
11          L_cls <- BCE(p, y)
12          L_att <- 0
13          if lambda_a > 0 then
14              q     <- m / sum(m)                        // rows with sum(m)>0
15              L_att <- - mean_i sum_t q_it * log(alpha_it)
16          L_ret <- 0
17          if lambda_r > 0 then
18              L_ret <- SUPCON(z, phi, kappa)
19          L <- L_cls + lambda_a * L_att + lambda_r * L_ret
20          theta <- ADAMW(theta, CLIP(grad L, 1.0), eta_t)
21      f1 <- EVALUATE(theta, VAL)
22      if f1 > best then best <- f1 ; theta* <- theta ; stale <- 0
23      else stale <- stale + 1 ; if stale >= patience then break
24  return theta*

ALGORITHM 2  FAMILY-BALANCED-SAMPLER
────────────────────────────────────────────────────────────────────────────
input :  D, batch size B, families per batch F
 1  repeat |D| / B times
 2      S <- sample F distinct families without replacement
 3      batch <- {}
 4      for each family f in S do
 5          batch <- batch  U  sample( floor(B/F) items of family f )
 6      yield batch
 ── rationale: L_ret needs several members of a family inside one batch,
    otherwise P(i) is empty and the contrastive term is degenerate.

ALGORITHM 3  Neural code search (inference)
────────────────────────────────────────────────────────────────────────────
input :  query function q, corpus C, trained theta*
 1  z_q <- NORMALISE(SEARCH-HEAD(POOL(ENCODER(q))))
 2  for each c in C do  z_c <- cached embedding of c
 3  s_c <- z_q . z_c                                  // cosine similarity
 4  return C sorted by s_c descending
 ── the corpus is embedded once; a query is one forward pass plus one matmul.
```

### 5.5 The ablation design

Each variant removes exactly one component, so every row of the results table
isolates one claim.

| Variant | Pooling | $\lambda_a$ | $\lambda_r$ | Isolates |
|---|---|---|---|---|
| M4 | mean | 0 | 0 | value of attention at all |
| M5 | attention | 0 | 0 | value of *guiding* the attention |
| M6 | attention | 0 | 0.3 | value of the contrastive search head |
| **M7 (proposed)** | attention | **0.5** | **0.3** | the full method |
""")

code(r'''
_m = AGNCS(vocab_size=len(vocab), d=D_MODEL, max_len=MAX_LEN)
print(_m)
print("\ntrainable parameters: {:,}".format(_m.n_params()))
print("receptive field: %d tokens" % (1 + 2 * sum((1, 2, 4))))
pd.DataFrame([{"variant": k, "description": VARIANT_LABELS[k], **v}
              for k, v in VARIANTS.items()])
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 6. Evaluation metrics

Write TP, FP, TN, FN for the confusion-matrix entries at threshold $\tau$, and
$N$ for the number of evaluated samples.

### 6.1 Detection

$$\text{Accuracy}=\frac{TP+TN}{N} \qquad
\text{Precision}=\frac{TP}{TP+FP} \qquad
\text{Recall}=\frac{TP}{TP+FN}$$

$$F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
\qquad
\textbf{FPR} = \frac{FP}{FP+TN}
\qquad
\text{MCC}=\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}}$$

**How to read them.** Recall is what a security team cares about first — a missed
vulnerability ships. **FPR is what decides whether the tool survives contact with
developers**, because every false alert spends review budget and erodes trust; it
is the headline metric of this study. MCC is reported because it stays
informative under class imbalance, taking $+1$ for perfect prediction, $0$ for
chance and $-1$ for total disagreement. ROC-AUC and PR-AUC summarise ranking
quality across all thresholds; they are **undefined for the rule-based
analysers**, which emit a hard decision rather than a score, and are shown as
`--` for those rows.

### 6.2 Code search

Relevance: a corpus item is relevant to a query if it belongs to the **same
vulnerability family** — it realises the same taint pattern.

$$\text{MRR} = \frac{1}{|Q|}\sum_{q}\frac{1}{\text{rank}_q} \qquad
\text{Recall@}k = \frac{1}{|Q|}\sum_q \mathbb{1}[\,\exists\text{ relevant item in top }k\,]$$

$$\text{MAP}=\frac{1}{|Q|}\sum_q \frac{1}{R_q}\sum_{j:\,\text{rel}_j} P@j
\qquad
\text{nDCG@}k=\frac{\sum_{j=1}^{k}\text{rel}_j/\log_2(j+1)}{\text{IDCG@}k}$$

MRR answers "how far must an analyst scroll before the first useful hit?"; MAP
and nDCG reward getting the *whole* ranking right, nDCG with an explicit
position discount.

### 6.3 Attention quality

This is the part the project title promises and most vulnerability-detection
papers omit: evidence that the model looks in the right place.

$$\text{AMR}_i = \sum_t \alpha_{i,t}\,m_{i,t}
\qquad
\textbf{Lift}_i = \frac{\text{AMR}_i}{|m_i| \,/\, |\text{valid}_i|}
\qquad
\text{Pointing}_i = \mathbb{1}\big[\arg\max_t \alpha_{i,t} \in m_i\big]$$

$$k_i = \lvert m_i \rvert, \quad
T_i = \operatorname*{arg\,top}_{k_i}\; \alpha_i, \quad
\text{IoU}_i = \frac{\lvert T_i \cap m_i \rvert}{\lvert T_i \cup m_i \rvert}$$

Here $T_i$ is the set of the $k_i$ most-attended tokens, where $k_i$ is the
number of annotated tokens, so the comparison is size-matched.

Lift divides out the mass a *uniform* map would place on the annotation, so
**Lift = 1 means "no better than looking everywhere"** and the number is
comparable across functions of different lengths. Pointing accuracy is the
strictest single-number version: does the single most-attended token lie on the
taint path?

### 6.4 Statistical protocol

Every neural variant is trained with 3 seeds. Tables report mean ± sd; claims of
improvement are backed by a **paired** $t$-test and a Wilcoxon signed-rank test
over the matched seeds, with Cohen's $d_z$ for effect size. Thresholds and early
stopping use `val` only; `test_id` and `test_ood` are read once, at the end.
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 7. Baselines

**M1 — pattern-matching analyser.** Reimplements the published rule family
`python.lang.security.audit.formatted-sql-query` (semgrep) / Bandit **B608**:
flag the query argument of an `execute`-like call whenever it is produced by
f-string interpolation, `%`-formatting, `str.format` or `+` concatenation. This
is what most CI-integrated tooling actually does.

**M2 — AST taint analyser.** A genuine intra-procedural forward taint analysis:
parameters and request objects are sources; assignments, string operations,
loop and comprehension targets and container accumulation propagate taint;
`int()`/`float()` are recognised sanitisers; `execute`-like calls are sinks,
including keyword forms such as `.extra(where=[...])`. It resolves variables to
their definitions, so it follows multi-hop and inter-procedural flows that M1
cannot see.

M2 is deliberately **not** a strawman — the comparison is only worth making
against a static analyser that is actually good.

**M3 — TF-IDF + logistic regression** over *the same token stream the neural
models receive*, so any neural gain is attributable to the architecture rather
than to better preprocessing. Its L2-normalised TF-IDF vectors double as a
classical code-search index.
""")

code(r'''
t0 = time.time()
static_df, static_preds = run_static_baselines(frames)
print("\nstatic analysers evaluated in %.1fs" % (time.time() - t0))
''')

code(r'''
t0 = time.time()
tfidf_df, tfidf_art = run_tfidf_baseline(frames, splits, seeds=SEEDS, n_train=N_TRAIN)
tfidf_ret = pd.DataFrame(tfidf_retrieval(tfidf_art, splits, seed=SEEDS[0]))
if len(tfidf_ret):
    tfidf_df = tfidf_df.merge(tfidf_ret, on=["method", "seed", "split"], how="left")
print("\nTF-IDF baseline fitted in %.1fs" % (time.time() - t0))
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 8. Training the neural variants

Four variants × `len(SEEDS)` seeds. Each epoch prints the training loss split
into its three components and the validation metrics, so the effect of the
guidance term is visible as it happens.
""")

code(r'''
t0 = time.time()
neural_df, art = run_neural(splits, len(vocab), MAX_LEN, variants=list(VARIANTS),
                            seeds=SEEDS, cfg_kwargs=CFG, n_train=N_TRAIN,
                            n_val=N_VAL, keep_alpha_seed=SEEDS[0])
print("\nall neural training finished in %.1f min" % ((time.time() - t0) / 60))
''')

code(r'''
all_df = pd.concat([static_df, tfidf_df, neural_df], ignore_index=True)
os.makedirs("results/tables", exist_ok=True)
all_df.to_csv("results/tables/all_runs.csv", index=False)

summary = summarise(all_df, MAIN_METRICS + SEARCH_METRICS + ATT_METRICS + ["vuln_p@10", "vuln_mrr"])
summary.to_csv("results/tables/summary_by_method_split.csv", index=False)

thresholds = {(r["method"], r["seed"]): r.get("threshold", 0.5)
              for _, r in all_df.iterrows() if not pd.isna(r.get("threshold"))}
methods = [m for m in METHOD_ORDER if m in set(all_df["method"])]
print("methods evaluated:", len(methods), "| runs:", len(all_df))
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 9. Results

### 9.1 Detection

`test_id` is the in-distribution test set. `test_ood` contains **six
vulnerability families the models never saw**. Read the two tables as a pair:
the first says whether the model fits the problem, the second says whether it
learned anything transferable.
""")

code(r'''
for sp in ("test_id", "test_ood"):
    t = pretty_table(summary, sp)
    t.to_csv("results/tables/table_detection_%s.csv" % sp, index=False)
    print("\n" + "=" * 118)
    print("Table: detection performance -- %s" % viz.SPLIT_TITLE[sp])
    print("=" * 118)
    display(t)
''')

code(r'''
fig = viz.fig_main_comparison(summary)
plt.show()
''')

code(r'''
# The TF-IDF probabilities live in `tfidf_art`, so merge both artefact dicts --
# passing `art` alone would silently drop that baseline's curve.
scored = {**art, **tfidf_art}
fig = viz.fig_roc_pr(scored, splits,
                     [m for m in methods if m.startswith(("M3", "M4", "M5", "M6", "M7"))],
                     seed=SEEDS[0], static_preds=static_preds)
plt.show()
''')

md(r"""
### 9.2 The metric that decides deployability

Overall FPR averages over easy safe code (plain parameterised queries, which
nothing gets wrong) and the hard negatives. Splitting them apart shows where the
false positives actually come from.

Note that the held-out split has **no easy negatives by construction** (§3.3),
so its `fpr_easy_negatives` column is empty — that is the design, not missing
data.
""")

code(r'''
hn = hard_negative_fpr(frames, static_preds, {**art, **tfidf_art}, methods, thresholds, SEEDS)
hn.to_csv("results/tables/hard_negative_fpr.csv", index=False)
pivot = (hn.groupby(["method", "split"])[["fpr_hard_negatives", "fpr_easy_negatives"]]
           .mean().round(4).reset_index())
pivot["method"] = pivot["method"].map(METHOD_LABELS)
display(pivot)
fig = viz.fig_hard_negative_fpr(hn)
plt.show()
''')

md(r"""
### 9.3 Where each method fails

The heatmap is the diagnostic view: one column per family, one row per method,
cell = error rate. It turns an aggregate score into a statement about *which
reasoning step* a method is missing.
""")

code(r'''
# Averaged over seeds: a single seed's per-family breakdown is not safe to read
# on the held-out split, where the seed-to-seed spread is large.
pf = per_family_table_multiseed(frames, static_preds, {**art, **tfidf_art},
                                methods, thresholds, SEEDS)
pf.to_csv("results/tables/per_family_errors.csv", index=False)
for sp in ("test_id", "test_ood"):
    fig = viz.fig_family_heatmap(pf, sp)
    plt.show()
''')

code(r'''
# The families that remain hardest, ranked by the proposed model's error rate.
cols = [m for m in METHOD_ORDER if m in set(pf.method)]
worst = (pf[pf.split == "test_ood"]
         .pivot_table(index="family", columns="method", values="error_rate")
         .reindex(columns=cols)
         .sort_values("M7_AGNCS_full", ascending=False))
display(worst.round(3))

# Same families, with the seed spread of the proposed model -- read the error
# rate and its spread together before concluding a family is solved or broken.
spread = (pf[(pf.split == "test_ood") & (pf.method == "M7_AGNCS_full")]
          .set_index("family")[["label", "error_rate", "error_sd", "n_seeds"]]
          .reindex(worst.index))
spread["error_rate +/- sd"] = [("%.3f +/- %.3f" % (r, s))
                               for r, s in zip(spread.error_rate, spread.error_sd)]
display(spread[["label", "error_rate +/- sd", "n_seeds"]])
''')

md(r"""
### 9.4 Attention quality and code search

These two subsections test the two halves of the project title.

**The right reference point for the attention metrics is M5, not uniform.**
Unguided attention (M5, $\lambda_a = 0$) already reaches a lift of roughly 2.7
on its own: the taint-carrying tokens are the predictive ones, so attention
drifts onto them without any supervision at all. The uniform line at 1.0 in the
figure is therefore a floor, not a baseline. The honest question is not "does
the guided model attend to the taint path?" — both attention models largely do
— but **how much the guidance adds on top of an already-informative attention
map**, which is the M7-minus-M5 difference, tested for significance in §13.
Claiming that guidance "makes the model look in the right place" would overstate
it; guidance *sharpens* a map that was already roughly right.
""")

code(r'''
for sp in ("test_id", "test_ood"):
    t = pretty_table(summary, sp, metrics=ATT_METRICS)
    t.to_csv("results/tables/table_attention_%s.csv" % sp, index=False)
    print("\nAttention quality -- %s" % viz.SPLIT_TITLE[sp]); display(t)
fig = viz.fig_attention_quality(summary)
plt.show()
''')

code(r'''
for sp in ("test_id", "test_ood"):
    t = pretty_table(summary, sp, metrics=SEARCH_METRICS)
    t.to_csv("results/tables/table_search_%s.csv" % sp, index=False)
    print("\nCode search (relevance = same vulnerability family) -- %s" % viz.SPLIT_TITLE[sp])
    display(t)
fig = viz.fig_retrieval(summary)
plt.show()
''')

code(r'''
# Training dynamics and the loss decomposition.
fig = viz.fig_training_curves(art, list(VARIANTS), seed=SEEDS[0]); plt.show()
fig = viz.fig_loss_components(art, "M7_AGNCS_full", seed=SEEDS[0]); plt.show()
''')

code(r'''
# Confusion matrices: best static analyser vs the proposed model.
pairs = []
for sp in ("test_id", "test_ood"):
    y = splits[sp].y
    pairs.append(("M2 static taint", viz.SPLIT_TITLE[sp], y,
                  (static_preds[("M2_static_taint", sp)] >= 0.5).astype(int)))
    thr = thresholds.get(("M7_AGNCS_full", SEEDS[0]), 0.5)
    pairs.append(("M7 AGNCS (proposed)", viz.SPLIT_TITLE[sp], y,
                  (art[("prob", "M7_AGNCS_full", SEEDS[0], sp)] >= thr).astype(int)))
fig = viz.fig_confusion(pairs)
plt.show()
''')

md(r"""
### 9.5 Receptive-field ablation

The §4.2 prediction, tested: if the encoder must span the taint path, then
widening the receptive field should help, and help most on the held-out
families whose spans are widest.

**It does not.** Across $R \in \{15, 31, 63, 127\}$:

* in-distribution F1 is flat (spread 0.0013 -- the task is saturated);
* held-out F1 varies by 0.121, roughly **90x** the in-distribution spread;
* the correlation runs the *wrong* way: wider $R$ gives a worse held-out FPR
  (Spearman $\rho = +0.80$) and worse held-out ROC-AUC ($\rho = -0.80$);
* the widest and the narrowest configurations bracket the best score, so the
  relationship is not even monotonic.

Two conclusions, the second more important than the first.

1. **The decisive evidence is local.** The quoted-interpolation motif sitting
   next to SQL keywords fits inside 15 tokens. A field wide enough to span the
   whole taint path adds no necessary information and gives the model room to
   fit family-specific long-range layout, which is exactly what fails to
   transfer.

2. **With a saturated validation signal, OOD generalisation is unselectable.**
   Validation F1 is ~0.999 for every configuration, so early stopping picks an
   essentially arbitrary epoch -- watch `best_epoch` jump around in the table
   below -- and the held-out score is then dominated by seed noise. This is one
   seed per configuration, so **the ablation is not evidence that a small field
   is better**; it is evidence that this axis cannot be tuned with the signal
   available. Choosing $R$ by its held-out score would be selecting on the test
   set. We therefore select on validation, where all four tie, and break the tie
   on cost: dilations $1,2,4$.

The same caution applies to every held-out number in this report, which is why
§9.1 reports them with a standard deviation over three seeds. Look at that width
before believing any single OOD comparison.

Run `python scripts/ablate_receptive_field.py` (about 17 minutes) if the file is
absent.

""")

code(r'''
rf_path = "results/tables/ablation_receptive_field.csv"
if os.path.exists(rf_path):
    rf = pd.read_csv(rf_path)
    display(rf[["dilations", "receptive_field", "n_params", "train_seconds",
                "best_epoch", "test_id_f1", "test_id_fpr", "test_ood_f1",
                "test_ood_fpr", "test_ood_recall", "test_ood_auc",
                "test_ood_att_lift"]])
    print("in-distribution F1 spread: %.4f   held-out F1 spread: %.4f  (ratio %.0fx)"
          % (rf.test_id_f1.max() - rf.test_id_f1.min(),
             rf.test_ood_f1.max() - rf.test_ood_f1.min(),
             (rf.test_ood_f1.max() - rf.test_ood_f1.min())
             / max(rf.test_id_f1.max() - rf.test_id_f1.min(), 1e-9)))
    for c in ("test_ood_fpr", "test_ood_auc"):
        print("Spearman rho(receptive_field, %-13s) = %+.2f"
              % (c, rf["receptive_field"].corr(rf[c], method="spearman")))

    spans = {}
    for sp in ("test_id", "test_ood"):
        st = splits[sp]
        spans[sp] = np.array([(lambda ix: ix[-1] - ix[0] + 1 if len(ix) else 0)(
            np.flatnonzero(st.mask[i])) for i in range(len(st))])
    fig = viz.fig_receptive_field(spans, rf)
    plt.show()
else:
    print("not run yet -- see scripts/ablate_receptive_field.py")
''')

md(r"""
### 9.6 Training-set-size ablation

The justification for `N_TRAIN`. If this file is absent, run
`python scripts/ablate_trainsize.py` (about 25 minutes).
""")

code(r'''
abl_path = "results/tables/ablation_train_size.csv"
if os.path.exists(abl_path):
    abl = pd.read_csv(abl_path)
    display(abl)
    fig = viz.fig_trainsize(abl); plt.show()
else:
    print("ablation not run yet -- see scripts/ablate_trainsize.py")
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 10. Neural code search in action

Two retrieval tasks, both run on the trained embedding.

**Task A — query by example.** Given a vulnerable function, retrieve others that
realise *the same* taint pattern. Relevance = same family. This is the metric
tabulated in §9.4.

**Task B — transfer search.** Query with a vulnerable snippet from a **held-out**
family and search a corpus drawn from the families seen in training. No
same-family match exists, so relevance is simply "is also vulnerable". This is
the operational use case: an analyst finds one new injection pattern and asks
the index for everything else in the codebase that is dangerous for the same
reason.
""")

code(r'''
xb_rows = []
for m in [v for v in VARIANTS if ("embed", v, SEEDS[0], "test_ood") in art]:
    r = eval_cross_pattern_retrieval(
        art[("embed", m, SEEDS[0], "test_ood")], splits["test_ood"],
        art[("embed", m, SEEDS[0], "test_id")], splits["test_id"])
    xb_rows.append({"method": METHOD_LABELS[m], **{k: round(v, 4) for k, v in r.items()}})
xb = pd.DataFrame(xb_rows)
xb.to_csv("results/tables/table_transfer_search.csv", index=False)
print("Task B -- OOD vulnerable query vs in-distribution corpus (relevance = is vulnerable)")
print("A random corpus sample would score 0.50 on every column.\n")
display(xb)
''')

code(r'''
# A concrete search: one query, its top-5 neighbours, as a reviewer would see it.
def search_demo(method="M7_AGNCS_full", split="test_ood", q_family="V13_join_fragments", k=5):
    E, S, F = art[("embed", method, SEEDS[0], split)], splits[split], frames[split]
    cand = np.flatnonzero((F.family == q_family).values & (S.y == 1))
    if len(cand) == 0:
        print("no query available"); return
    qi = int(cand[0])
    corpus = np.flatnonzero(np.arange(len(S)) != qi)
    sim = E[corpus] @ E[qi]
    top = corpus[np.argsort(-sim)[:k]]

    print("#" * 100)
    print("QUERY   family=%s   label=%d" % (F.family.iloc[qi], S.y[qi]))
    print("#" * 100)
    print(F.code.iloc[qi])
    for rank, j in enumerate(top, 1):
        print("\n" + "-" * 100)
        print("HIT %d   cosine=%.4f   family=%s   label=%d   %s"
              % (rank, float(E[j] @ E[qi]), F.family.iloc[j], S.y[j],
                 "SAME PATTERN" if F.family.iloc[j] == q_family else ""))
        print("-" * 100)
        print(F.code.iloc[j])

search_demo()
''')

code(r'''
# The learned search space.
try:
    embs = {m: art[("embed", m, SEEDS[0], "test_id")]
            for m in ("M5_CNN_attention", "M7_AGNCS_full")
            if ("embed", m, SEEDS[0], "test_id") in art}
    fig = viz.fig_embedding_space(embs, splits["test_id"].fam, splits["test_id"].y, [], n=2000)
    plt.show()
except Exception as e:
    print("t-SNE skipped:", e)
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 11. Attention case study

Side by side: where the unguided model looks, and where the guided model looks,
on the same function. Orange bars are the annotated taint path.
""")

code(r'''
models_for_viz = {k: art[("model", k, SEEDS[0])] for k in ("M5_CNN_attention", "M7_AGNCS_full")
                  if ("model", k, SEEDS[0]) in art}

def case_study(split, family, idx=0, name=None):
    F = frames[split]
    sub = F[(F.family == family)]
    if len(sub) <= idx:
        return
    r = sub.iloc[idx]
    fig = viz.fig_attention_lines(
        models_for_viz, r.code, vocab, MAX_LEN, parse_rationale(r.rationale_lines),
        title="%s  (%s, %s)" % (family, "vulnerable" if r.label else "safe", split),
        out=name or ("fig7_attention_%s.png" % family))
    plt.show()

case_study("test_id", "V05_multi_hop_alias")
case_study("test_id", "S07_allowlisted_identifier")
case_study("test_ood", "V13_join_fragments")
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 12. Sanity check on hand-written code

Everything so far was generated by the same grammar that produced the training
data. These twelve snippets were written by hand, outside that grammar, in the
style of real application code (5 vulnerable, 7 safe). They are the closest thing
in this notebook to an external test set, and the honest place to look for the
synthetic-data gap discussed in §14.

The static analysers are the useful control here, because they have no training
data and so cannot be advantaged or disadvantaged by the grammar. Their failure
modes on these hand-written snippets **match the ones they show on the synthetic
corpus**: the pattern-matching analyser falsely flags the integer cast and the
allow-listed `ORDER BY`, and the taint analyser clears the cast but still flags
the allow-list. That agreement is evidence that the corpus reproduces the right
*difficulty structure* even though it is synthetic — which is the single most
important thing to establish about it. Compare the neural models' numbers below
against that reference point rather than against each other.
""")

code(r'''
HAND = [
 (1, "classic f-string into execute", """
def find_user(conn, email):
    cur = conn.cursor()
    cur.execute(f"SELECT id, name FROM accounts WHERE email = '{email}'")
    return cur.fetchone()
"""),
 (0, "parameterised equivalent", """
def find_user(conn, email):
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM accounts WHERE email = ?", (email,))
    return cur.fetchone()
"""),
 (0, "integer cast then interpolated", """
def get_order(conn, order_id):
    oid = int(order_id)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM orders WHERE id = {oid}")
    return cur.fetchall()
"""),
 (1, "ORDER BY identifier interpolated, values bound", """
def list_products(conn, category, sort_by):
    cur = conn.cursor()
    sql = f"SELECT sku, price FROM products WHERE category = %s ORDER BY {sort_by}"
    cur.execute(sql, (category,))
    return cur.fetchall()
"""),
 (0, "allow-listed ORDER BY", """
ALLOWED = {"price", "sku", "created_at"}

def list_products(conn, category, sort_by):
    if sort_by not in ALLOWED:
        raise ValueError("bad sort column")
    cur = conn.cursor()
    sql = f"SELECT sku, price FROM products WHERE category = %s ORDER BY {sort_by}"
    cur.execute(sql, (category,))
    return cur.fetchall()
"""),
 (1, "useless escaping before interpolation", """
def search(conn, term):
    safe = term.replace(";", "")
    cur = conn.cursor()
    cur.execute("SELECT * FROM docs WHERE title LIKE '%" + safe + "%'")
    return cur.fetchall()
"""),
 (0, "LIKE wildcards passed as a bound parameter", """
def search(conn, term):
    cur = conn.cursor()
    cur.execute("SELECT * FROM docs WHERE title LIKE ?", (f"%{term}%",))
    return cur.fetchall()
"""),
 (1, "taint through a helper function", """
def _where(value):
    return f"WHERE username = '{value}'"

def load(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users " + _where(username))
    return cur.fetchall()
"""),
 (0, "SQLAlchemy bound parameters", """
def load(session, username):
    stmt = text("SELECT * FROM users WHERE username = :u").bindparams(u=username)
    return session.execute(stmt).fetchall()
"""),
 (0, "SQL-looking string that is never executed", """
def describe_filter(column, value):
    preview = f"SELECT * FROM audit WHERE {column} = '{value}'"
    return {"preview": preview, "note": "rendered for the UI, never executed"}
"""),
 (1, "request dictionary straight into the query", """
def handler(conn, request):
    term = request.get_json()["q"]
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM items WHERE name = '{term}'")
    return cur.fetchall()
"""),
 (0, "ORM query builder", """
def handler(session, term):
    return session.query(Item).filter(Item.name == term).all()
"""),
]

from src.baselines import static_regex_predict, static_taint_predict
from src.train import predict as nn_predict
from src.data import EncodedSplit
from src.code_tokenizer import tokenise as _tok
from src.data import truncate_head_tail

def encode_snippets(snips):
    n = len(snips)
    X = np.full((n, MAX_LEN), vocab.pad_id, dtype=np.int32)
    V = np.zeros((n, MAX_LEN), dtype=bool)
    for i, c in enumerate(snips):
        ids = vocab.encode(truncate_head_tail(_tok(c)[0], MAX_LEN))
        X[i, :len(ids)] = ids; V[i, :len(ids)] = True
    return EncodedSplit(x=X, mask=np.zeros_like(X, dtype=np.uint8), valid=V,
                        y=np.zeros(n, np.int8), fam=np.zeros(n, np.int16),
                        hard_neg=np.zeros(n, np.int8), index=np.arange(n, dtype=np.int32))

ys    = np.array([h[0] for h in HAND])
names = [h[1] for h in HAND]
codes = [h[2].strip("\n") for h in HAND]
enc   = encode_snippets(codes)

rows = []
for i in range(len(HAND)):
    rows.append({"#": i, "snippet": names[i], "truth": "VULN" if ys[i] else "safe",
                 "M1 regex": static_regex_predict(codes[i]),
                 "M2 taint": static_taint_predict(codes[i])})
res = pd.DataFrame(rows)

for m in [v for v in VARIANTS if ("model", v, SEEDS[0]) in art]:
    p = nn_predict(art[("model", m, SEEDS[0])], enc, want_embed=False)["prob"]
    thr = thresholds.get((m, SEEDS[0]), 0.5)
    res[m.split("_")[0] + " p"] = np.round(p, 3)
    res[m.split("_")[0]] = (p >= thr).astype(int)

display(res)
acc = {c: float((res[c].values == ys).mean()) for c in res.columns
       if c in ("M1 regex", "M2 taint") or (len(c) == 2 and c.startswith("M"))}
print("\naccuracy on the 12 hand-written snippets:")
for k, v in acc.items():
    print("   %-10s %.3f  (%d/%d)" % (k, v, int(v * len(ys)), len(ys)))
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 13. Statistical significance

Paired over the matched seeds: the proposed model against the two ablations it
is supposed to beat. A difference is only worth reporting if it survives this.
""")

code(r'''
sig = pd.concat([
    significance_table(all_df, "M5_CNN_attention", "M7_AGNCS_full").assign(comparison="M7 vs M5 (guidance + search head)"),
    significance_table(all_df, "M6_CNN_attn_contrastive", "M7_AGNCS_full").assign(comparison="M7 vs M6 (guidance only)"),
    significance_table(all_df, "M4_CNN_meanpool", "M7_AGNCS_full").assign(comparison="M7 vs M4 (attention at all)"),
], ignore_index=True)
sig.to_csv("results/tables/significance_tests.csv", index=False)
display(sig[["comparison", "split", "metric", "proposed", "baseline", "diff",
             "t_p", "wilcoxon_p", "cohen_dz", "n_seeds"]])
print("\nNote: with %d seeds the Wilcoxon test cannot reach p < 0.25, so the paired" % len(SEEDS))
print("t-test and Cohen's d_z carry the evidence; treat p-values as indicative.")
''')

# --------------------------------------------------------------------------- #
md(r"""
---
## 14. Limitations and conclusions

### What the experiment supports

Read §9 before this. The results are **mixed, and the project's two halves come
apart** — the search half succeeds, the detection half does not beat a classical
baseline. Reporting that split honestly is the main finding.

**1. The in-distribution split cannot discriminate between methods at all.**
TF-IDF + logistic regression scores a *perfect* 1.0000 on F1, FPR, recall,
precision, ROC-AUC and MCC over 10,392 in-distribution samples. The synthetic
corpus is linearly separable in token-n-gram space within the families it was
built from. Every neural variant scores 0.995-0.999. Nothing in that column
supports any claim, and it should be reported as a property of the corpus rather
than as a result.

**2. On held-out families, the classical baseline wins the detection task.**

| | held-out F1 | held-out FPR | held-out ROC-AUC |
|---|---|---|---|
| M3 TF-IDF + LR | **0.829** | 0.349 | **0.930** |
| M4 CNN, mean pooling | 0.714 | **0.079** | 0.788 |
| M5 CNN + unguided attention | 0.642 | 0.793 | 0.567 |
| M6 CNN + attention + contrastive | 0.536 | 0.908 | 0.370 |
| M7 AGNCS (proposed) | 0.600 | 0.325 | 0.595 |

A bag-of-token-n-gram model generalises to unseen injection patterns
substantially better than any variant of the proposed architecture. This is
consistent with §9.5: if the decisive evidence is a short local motif, a lexical
model captures it directly and has no long-range capacity with which to overfit.

**3. Attention *hurts* out-of-distribution detection here.** The attention-free
control (M4) is the best neural detector on held-out families by F1, FPR and
AUC. Adding unguided attention (M5) raises held-out FPR from 0.079 to 0.793;
adding the contrastive head on top (M6) pushes it to 0.908 with ROC-AUC 0.370,
which is *worse than chance*. The proposed model does not overturn this: it is
beaten by plain mean pooling.

**4. What attention guidance demonstrably does do** (M7 vs M6, paired over 3
seeds) is repair the damage the search head inflicts on detection, and sharpen
the attention map:

| metric | M6 | M7 | diff | p | Cohen's d_z |
|---|---|---|---|---|---|
| held-out FPR | 0.908 | 0.325 | **-0.583** | 0.011 | -5.42 |
| in-dist. attention lift | 2.26 | 3.30 | **+1.03** | 0.0015 | 15.07 |
| held-out attention lift | 1.88 | 2.36 | +0.47 | 0.038 | 2.90 |
| held-out F1 | 0.536 | 0.600 | +0.064 | 0.35 | 0.71 |

So the defensible claim is narrow and specific: **$\lambda_a$ acts as a
regulariser that makes the contrastive search objective compatible with
detection.** Without guidance, bolting a retrieval head onto the encoder
destroys held-out detection (AUC 0.370). With it, near-perfect retrieval is
retained *and* detection returns to above chance. The held-out F1 gain itself is
**not** statistically significant (p = 0.35) and must not be claimed.

**5. The code-search half of the project succeeds clearly.** Relevance = same
vulnerability family:

| | in-dist. MAP | held-out MAP | held-out MRR |
|---|---|---|---|
| M3 TF-IDF cosine | 0.387 | 0.796 | 0.995 |
| M4 mean pooling | 0.178 | 0.418 | 0.885 |
| M5 unguided attention | 0.288 | 0.383 | 0.941 |
| M6 attention + contrastive | 0.996 | 0.861 | 0.988 |
| M7 AGNCS (proposed) | **0.996** | **0.912** | 0.996 |

The contrastive term is what creates a usable index (MAP 0.18-0.29 without it,
0.996 with it), and guidance adds a further 0.05 MAP on held-out families. This
is the one place the proposed model beats every baseline on both splits, and it
is a capability the static analysers do not have at all.

**6. Attention guidance works as a localisation mechanism**, independently of
whether it improved accuracy: in-distribution top-k IoU with the annotated taint
path rises from 0.506 (unguided) to **0.971**, pointing-game accuracy from 0.912
to 0.996, with $KL(q\,\|\,\alpha)$ converging to 0.084. The mechanism does what
it was designed to do.

**7. The static analysers behave as the motivation predicts, and differ sharply
from each other.** M2 (AST taint) achieves **recall 1.000 on both splits** — it
misses nothing, including all three held-out vulnerability families — at FPR
0.167 in-distribution and 0.667 on held-out safe code, all of it on hard
negatives. M1 (pattern rules) has recall 0.980 in-distribution but **0.147** on
held-out families: its rules enumerate surface forms instead of reasoning about
flow. Do not describe the two as behaving alike.

### Overall verdict

The project's premise is confirmed (static analysis is high-recall and
false-positive-prone, concentrated entirely on hard negatives) and its proposed
remedy is only **partly** supported. Attention-guided neural code *search* is a
clear success. Attention-guided *detection* does not beat TF-IDF on unseen
vulnerability patterns, and attention itself is harmful to out-of-distribution
detection in this setting. The practical recommendation that follows is a
pipeline rather than a replacement: M2 taint analysis as the high-recall first
pass, and the AGNCS embedding as the retrieval-and-ranking layer that decides
what a human reviews first — which uses each component for the thing it is
actually good at.

### Limitations, stated plainly

* **The corpus is synthetic.** This is the dominant threat to validity. The
  grammar was written by the same author as the model, so it cannot contain
  injection idioms neither the author nor the generator thought of. The
  held-out-family split and the hand-written snippets in §12 mitigate this but
  do not remove it. Absolute numbers here should be read as *relative* evidence
  between methods, not as an estimate of field performance.
* **Function-level, intra-procedural scope.** Taint that crosses module
  boundaries, flows through class attributes, or arrives via a framework's
  dependency injection is outside what a single function shows. Real audits need
  whole-program analysis.
* **Annotation and model share an author.** The guidance target encodes one
  specific view of what the "relevant" statements are. A different annotator
  might mark the sanitiser rather than the sink.
* **Model selection had no OOD signal.** The validation split shares its
  families with training, so nothing in the selection loop could see
  generalisation to new patterns. The recommendation below fixes this.
* **Three seeds** is enough for a mean ± sd and a paired t-test; it is not
  enough for the Wilcoxon test to reach conventional significance, as §13 notes.
* **Attention is not a causal explanation.** High attention mass on the taint
  path shows the pooling layer weights those tokens; it does not prove the
  decision depended on them. A gradient-based or occlusion-based attribution
  check would be the next step.
* **No comparison against a pretrained code model.** CodeBERT or GraphCodeBERT
  fine-tuning is the obvious strong baseline; it was out of reach of the CPU
  budget (§5.1) and its absence should be counted against the strength of the
  conclusions.

### Recommended next steps

1. **Add a disjoint held-out-family *validation* split.** This is the direct fix
   for the selection problem above, and it is the change I would make first.
   Partition the 30 families three ways instead of two: ~22 for training, 4 whose
   families are held out and used *only* for model selection (`val_ood`), and 4
   reserved for reporting (`test_ood`). Early stopping and the decision threshold
   would then be tuned against a legitimate generalisation signal instead of a
   saturated one, and the receptive field (and every other architectural choice)
   could be tuned honestly rather than fixed by a cost tie-breaker. The current
   two-way split cannot do this, which is why the architecture ablation in §9.5
   had to be reported as inconclusive.

2. Validate on mined real-world Python CVEs (the SARD Python subset, or
   `PyPI`-sourced advisories) — the single most valuable *external* addition.
3. Replace the dilated CNN with a fine-tuned GraphCodeBERT given GPU budget, and
   test whether attention guidance still adds anything on top of pretraining.
4. Feed the model a program-dependence graph rather than a token sequence, so
   the aliasing the tokeniser currently *hints* at becomes explicit.
5. Use the retrieval head for its real purpose: an incremental index over a
   production codebase, re-ranked by the detection head.
""")

code(r'''
# Environment record and artefact manifest for the report.
import sklearn, scipy, matplotlib
env = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "processor": platform.processor(), "cpu_threads": torch.get_num_threads(),
    "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "config": {"QUICK": QUICK, "MAX_LEN": MAX_LEN, "N_TRAIN": N_TRAIN,
               "SEEDS": SEEDS, **CFG},
    "corpus": {k: stats[k] for k in ("n_total", "n_families", "splits")},
}
os.makedirs("results/raw", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
with open("results/raw/environment.json", "w", encoding="utf-8") as fh:
    json.dump(env, fh, indent=2)

if ("model", "M7_AGNCS_full", SEEDS[0]) in art:
    torch.save(art[("model", "M7_AGNCS_full", SEEDS[0])].state_dict(),
               "results/models/agncs_seed%d.pt" % SEEDS[0])

print(json.dumps(env, indent=2))
print("\nfigures written:")
for f in sorted(os.listdir("results/figures")):
    print("   results/figures/%s" % f)
print("\ntables written:")
for f in sorted(os.listdir("results/tables")):
    print("   results/tables/%s" % f)
''')

# =========================================================================== #

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0",
                          "mimetype": "text/x-python",
                          "file_extension": ".py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(root, NB)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("wrote %s  (%d cells: %d markdown, %d code)"
      % (NB, len(cells),
         sum(c["cell_type"] == "markdown" for c in cells),
         sum(c["cell_type"] == "code" for c in cells)))
