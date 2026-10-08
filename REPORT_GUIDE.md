# Mapping the build onto the NCOS734 report template

Which artefact answers each bracketed instruction in
*NCOS734 – Mini Project – REPORT TEMPLATE – 2026*. Marks in brackets.

Run `python scripts/run_experiments.py` (or the notebook) first; every path
below is produced by that run.

---

## Abstract — [12]

| Template line | Source |
|---|---|
| Background | Static analysers carry a high false-positive rate on SQL-injection rules; §7 of the notebook quantifies it for two analyser classes |
| Aim | Test whether supervising *where* a neural model attends reduces that false-positive rate and transfers to unseen injection patterns |
| Method | AGNCS: dilated-CNN encoder + multi-head additive attention pooling, trained with detection + attention-guidance + supervised-contrastive losses; 7 methods, 3 seeds, 60,000-sample taint-annotated corpus, 6 families held out |
| Findings | `results/tables/table_detection_test_id.csv`, `..._test_ood.csv`, `hard_negative_fpr.csv` |
| Conclusion | Notebook §14 |

Keep to 300–350 words. Quote FPR and F1 on **both** test splits — the ID/OOD
contrast is the result.

---

## Introduction — [12] (400–500 words)

* **Background / why it matters** — CWE-89 is still in the OWASP Top 10;
  developer trust in a scanner collapses once the alert queue is mostly noise.
* **What exists in the literature** — cite, at minimum:
  * Bandit / semgrep rule-based detection (the M1 class).
  * Static taint analysis for injection flaws (the M2 class).
  * Li et al., *VulDeePecker* (2018) — RNNs over code gadgets.
  * Zhou et al., *Devign* (2019) — GNNs on code property graphs.
  * Feng et al., *CodeBERT* (2020); Guo et al., *GraphCodeBERT* (2021).
  * Husain et al., *CodeSearchNet* (2019) — neural code search.
  * Khosla et al., *Supervised Contrastive Learning* (2020) — the `L_ret` term.
  * Bahdanau et al. (2015) — the additive attention being supervised.
  * Work on attention/rationale supervision (e.g. DeYoung et al., *ERASER*, 2020).
  * Jain & Wallace, *Attention is not Explanation* (2019) — cite this honestly
    when discussing the attention metrics, and see notebook §14.
* **What you set out to do** — the three-signal model, the ablation design, and
  the held-out-family protocol.
* **Who benefits** — security engineers triaging scanner output, and
  code-review tooling that must rank findings rather than dump them.
* **Report outline** — one sentence per section.

---

## Description of the problems solved — [20] (2 pages max)

Notebook **§2** is written to be transcribed almost directly.

* The two coupled tasks (detection $f_\theta$, search $g_\theta$) — §2.1.
* **Mathematical model** — the taint-reachability formulation in §2.2:
  $\exists v_0 \in S, v_n \in K$ with a path in $E^*$ avoiding $N$. Include the
  two-line `execute` comparison; it makes the whole difficulty concrete in four
  lines of code.
* **Variables and parameters** — the table in §2.4 ($x, y, \phi, m, H, \alpha,
  c, z, \lambda_a, \lambda_r, \tau$).
* **Relevance to AI** — §2.3: deciding membership of that set is a reachability
  *search*; classical analysis does it symbolically, this project learns it.
  That is the link to the brief's "Searching" area.
* **Diagram** — the architecture block diagram in §5.1.

---

## Description of the algorithms implemented — [15] (2 pages max)

Notebook **§5**.

* **Pseudocode** — §5.4 contains three fully indented listings: Algorithm 1
  (AGNCS training), Algorithm 2 (family-balanced sampler), Algorithm 3 (neural
  code search at inference). Paste them as-is.
* **How the algorithms work + all variables** — §5.2 (attention pooling
  equations) and §5.3 (the three losses, with the full `L_cls`, `L_att`,
  `L_ret` definitions and the combined objective).
* **Why these design choices** — the CNN-vs-Transformer throughput argument in
  §5.1 (96 vs 376 samples/s on the target CPU) and the rationale for
  family-balanced batching in Algorithm 2.
* **Baseline algorithms** — §7 describes M1 (semgrep/Bandit B608 rule) and M2
  (intra-procedural forward taint analysis with sources, propagation,
  sanitisers, sinks). Source: `src/baselines.py`.
* **Relevance to AI** — attention-based representation learning, metric
  learning, and learned search over a semantic embedding space.

---

## Materials and methods — [20] (2 pages max)

* **Experimental setup and parameter settings** — the config cell in notebook
  §1 and `results/raw/run_config.json`. Report: `MAX_LEN=160`, `D_MODEL=96`,
  dilations $1,2,4$ (receptive field 15), 4 attention heads, embedding dim 96,
  AdamW, lr $2.5\times10^{-3}$ with linear warm-up + cosine decay, weight decay
  $10^{-4}$, gradient clipping 1.0, batch 128, 16 families per batch,
  $\kappa=0.1$, $\lambda_a=0.5$, $\lambda_r=0.3$, early stopping patience 2,
  seeds $\{0,1,2\}$.
* **Dataset acquisition and description** — notebook §3 plus
  `data/corpus_stats.json` and `data/family_summary.csv`. State plainly:
  the corpus is **synthetic**, and say why (the table in §3.1 showing that
  Juliet/SARD, Devign, Big-Vul and CodeSearchNet each lack one of the three
  required annotations). Include the 30-family inventory table and the split
  table. Report the `ast.parse` validity rate (100%) and the
  anonymisation-leakage check from §4.1 (0.14% ID, 0.00% OOD).
* **Performance evaluation criteria** — notebook §6 gives every formula with its
  variables interpreted and a sentence on how to read each one. Cover all three
  groups: detection (incl. **FPR** as the headline and MCC for imbalance),
  code search (MRR, Recall@k, MAP, nDCG@k), attention quality (mass, **lift
  over uniform**, pointing, IoU). Note that ROC/PR-AUC are undefined for the
  rule-based analysers.
* **Statistical protocol** — 3 seeds, mean ± sd, paired *t*-test + Wilcoxon +
  Cohen's $d_z$; thresholds and early stopping on `val` only.
* **Software and hardware** — `results/raw/environment.json` records Python,
  OS, CPU, thread count and the exact versions of torch / numpy / pandas /
  scikit-learn / scipy / matplotlib.

---

## Discussion of results — [50] (3 pages max)

The heaviest-weighted section. Suggested order, with the artefact for each:

| # | Point to make | Table / figure |
|---|---|---|
| 1 | Static analysers: high recall, high FPR — the premise holds | `table_detection_*.csv`, `fig1`, `fig2` |
| 2 | ID detection is near-saturated for every learned model; say so rather than overselling it | `table_detection_test_id.csv` |
| 3 | The held-out-family split separates the methods | `table_detection_test_ood.csv`, `fig1` |
| 4 | **Where the false positives actually come from** | `hard_negative_fpr.csv`, `fig11` |
| 5 | Which reasoning step each method lacks | `per_family_errors.csv`, `fig5_*` |
| 6 | Attention guidance changes where the model looks | `table_attention_*.csv`, `fig6`, `fig7_*` |
| 7 | Code search quality, both tasks | `table_search_*.csv`, `table_transfer_search.csv`, `fig8`, `fig9` |
| 8 | Training dynamics and the loss decomposition | `fig3`, `fig3b` |
| 9 | Confusion matrices, best static vs proposed | `fig4` |
| 10 | **The receptive-field hypothesis and its refutation** | `taint_span_stats.csv`, `ablation_receptive_field.csv`, `fig12` |
| 11 | Why `N_TRAIN` is what it is | `ablation_train_size.csv`, `fig10` |
| 12 | Significance of the headline differences | `significance_tests.csv` |

Interpret, do not narrate. For each table say what the number *means* for
someone deploying the tool, and name the mechanism when a method fails — the
per-family heatmap exists so you can do exactly that.

---

## Critical evaluation — [5] (300–400 words)

Compare the methods against each other on their own terms:

* M1 vs M2 — what taint analysis buys over pattern matching, and the residual
  FPR it cannot remove because allow-lists and validators need semantic
  reasoning.
* M3 vs M4 — whether a learned encoder beats bag-of-token-n-grams on the same
  preprocessing.
* M4 vs M5 — whether attention pooling alone helps.
* M5 vs M6 — what the contrastive search head contributes.
* M6 vs M7 — the isolated contribution of **attention guidance**, which is the
  project's actual claim.
* Cost: M1/M2 need no training and run in sub-millisecond time per function;
  the neural models need training and a GPU to scale. Say when each is the right
  choice.

Then spend a paragraph on the **methodological** finding, which is the strongest
original content in the report and the kind of thing markers reward:

* In-distribution validation F1 saturates at ~0.999, so it carries almost no
  signal for model selection. Early stopping therefore picks an arbitrary epoch
  and held-out performance is dominated by seed noise — held-out F1 varied by
  0.121 across receptive-field settings against 0.0013 in-distribution.
* The consequence is that the receptive field could **not** be tuned honestly:
  choosing it by the held-out score would be selecting on the test set, so it was
  chosen on validation (a four-way tie) with cost as the tie-breaker.
* State plainly that the receptive-field ablation is *inconclusive about which
  field is better*. Resisting the temptation to present
  `ρ = +0.80` as a finding is the right call, and saying so earns more credit
  than a confident over-claim would.
* This is also why every held-out number is reported as mean ± sd over three
  seeds rather than as a single figure.

---

## Conclusion and recommendation — [5] (300–400 words)

Notebook §14 covers both halves. Be specific about:

* what the experiment supports, in order of strength;
* the synthetic-corpus limitation, stated plainly as the dominant threat to
  validity;
* the fact that in-distribution validation carried almost no signal for OOD
  generalisation, and what that costs the conclusions;
* usage recommendation: taint analysis as the high-recall first pass, the
  neural model as the ranker that decides what a human reviews first.

Recommendations the findings actually license — use these, they are specific:

1. **A disjoint held-out-family validation split** (~22 train / 4 `val_ood` for
   selection / 4 `test_ood` for reporting). The direct fix for the selection
   problem and the first change to make: without it no architectural choice can
   be tuned honestly, which is exactly why the receptive-field ablation had to
   be reported as inconclusive.
2. External validation on mined real Python CVEs (SARD Python subset, PyPI
   advisories).
3. GraphCodeBERT fine-tuning with GPU budget, to test whether attention
   guidance still adds anything on top of pretraining.
4. Program-dependence-graph input instead of a token sequence, making explicit
   the aliasing the tokeniser currently only hints at.
5. Gradient-based or occlusion attribution to check the attention claim, since
   attention mass is not by itself a causal explanation.
6. The retrieval head as an incremental index over a real codebase, re-ranked
   by the detection head.

---

## References — [5]

Pick one style and hold it. Cite only what you actually cite in the text. The
Introduction list above is the minimum set; add the SQL-injection and
static-analysis sources you use for the background paragraph.

---

## Notebook — [60]

`NCOS734_PG4_Attention_Guided_Neural_Code_Search.ipynb` runs top to bottom with
**Run All** and regenerates the corpus if `data/` is absent.

Before submitting:

1. Run `python scripts/verify_submission.py` and make sure nothing FAILS.
2. Run the notebook end to end with `QUICK = False` and **save the executed outputs** —
   the marker runs the code, but stored output protects you if their environment
   differs.
3. Check `pip install -r requirements.txt` from clean.
4. Put your names, student numbers, department and lecturer on the report title
   page (the template's placeholders).
5. Submit the whole project folder so `src/` resolves; the notebook adds the
   project root to `sys.path` itself.
