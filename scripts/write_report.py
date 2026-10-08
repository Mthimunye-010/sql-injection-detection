"""NCOS734 PG4 mini-project report -- content and assembly.

    python scripts/write_report.py     ->  NCOS734_PG4_Report.docx
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docx import Document

from build_report import (BODY_PT, EM, EN, OUT, bullet, caption, code, eq,
                          figure, heading, pagebreak, para, rich, setup, sub,
                          table)

doc = Document()
setup(doc)

TITLE = ("Attention-Guided Neural Code Search for Automated SQL Injection "
         "Vulnerability Detection in Python Source Code")

# ======================================================= TITLE PAGE
para(doc, "NCOS734 Semester Project", size=16, bold=True, align="center",
     before=72, after=12)
para(doc, TITLE, size=14, bold=True, align="center", after=12)
para(doc, "Session 2026", size=12, align="center", after=36)
for n in ("1st", "2nd", "3rd", "4th", "5th"):
    para(doc, "[ Insert %s Group Member name (Student Number) here ]" % n,
         align="center", after=5)
para(doc, "[ Insert Department, Faculty and University here ]", align="center",
     before=30, after=5)
para(doc, "Module Lecturer: MA Arasomwan", align="center", before=20, after=5)
para(doc, "October 05, 2026", align="center", before=20)
pagebreak(doc)

# ======================================================= ABSTRACT
heading(doc, "Abstract", "12")
rich(doc, [("Background: ", True, False),
 ("SQL injection (CWE-89) remains among the most damaging classes of web-application "
  "vulnerability. Static analysers detect it by matching syntactic patterns and are widely "
  "deployed in continuous-integration pipelines, but their output is dominated by false "
  "positives: safe code that merely resembles a dangerous construct " + EM + " an integer cast, an "
  "allow-listed column name, a regex-validated token " + EM + " trips the same rules. The effort "
  "spent discarding them, not any shortfall in recall, is the dominant cost of such "
  "tools.", False, False)])
rich(doc, [("Aim: ", True, False),
 ("This project tests whether supervising a neural model's attention " + EM + " training it to "
  "attend to the statements that carry the taint flow " + EM + " lowers the false-positive "
  "rate relative to static analysis, and whether the representation supports neural code "
  "search over vulnerability patterns.", False, False)])
rich(doc, [("Method: ", True, False),
 ("A corpus of 60,000 unique, valid Python functions was generated across 30 vulnerability "
  "and safety families, each annotated at statement level with its taint path or safety "
  "evidence, with six families withheld from training to measure generalisation. AGNCS, a "
  "dilated convolutional encoder with multi-head additive attention pooling, was trained "
  "jointly on detection, attention guidance and supervised contrastive retrieval. Seven "
  "methods were compared over three seeds.", False, False)])
rich(doc, [("Findings: ", True, False),
 ("The premise was confirmed: the AST taint analyser achieved recall 1.000 on both test "
  "splits while carrying a false-positive rate of 0.167 in-distribution and 0.667 on held-out "
  "safe code, concentrated entirely on hard negatives. The proposed remedy was only partly "
  "supported. On held-out families TF-IDF attained F1 0.829 against AGNCS's 0.600, and an "
  "attention-free control generalised better than any attention variant. Guidance nevertheless "
  "produced two significant effects: held-out false positives fell from 0.908 to 0.325 "
  "(p = 0.011) and the overlap between the attention map and the annotated taint path rose "
  "from 0.506 to 0.971. Code search reached 0.912 mean average precision against 0.796 for the "
  "strongest baseline.", False, False)])
rich(doc, [("Conclusion: ", True, False),
 ("Attention-guided neural code search succeeds as a retrieval and ranking mechanism but does "
  "not outperform a lexical baseline at detection on unseen injection patterns. The deployable "
  "configuration is therefore a pipeline rather than a replacement: taint analysis supplies "
  "recall, and the guided embedding supplies the ordering that decides what a human reviews "
  "first.", False, False)])

# ======================================================= INTRODUCTION
heading(doc, "Introduction", "12")
para(doc, "SQL injection arises when untrusted input is concatenated into a database query "
 "instead of being bound as a parameter. Catalogued as CWE-89 and still present in the OWASP "
 "Top Ten under Injection [1], [2], it is conceptually well understood and nevertheless "
 "persistent, because the defective and the correct idiom differ by only a few characters of "
 "source text.")
para(doc, "The tools deployed against it are predominantly static. Rule-based scanners such as "
 "Bandit's B608 check and the corresponding semgrep rule family flag any query argument "
 "constructed by string formatting or concatenation [3], [4]. These rules are cheap and "
 "transparent, but they reason about surface form rather than data flow, so they cannot "
 "distinguish interpolation of an attacker-controlled string from interpolation of an integer "
 "cast or of a column name checked against an allow-list. The consequence is an alert queue "
 "that developers learn to ignore " + EM + " a failure of precision, not of recall.")
para(doc, "Learning-based detection has been pursued as an alternative. VulDeePecker introduced "
 "recurrent models over sliced code gadgets [5]; Devign applied graph neural networks to code "
 "property graphs [6]; Big-Vul supplied large-scale mined labels [7]. Transformer models "
 "pre-trained on source code, notably CodeBERT and GraphCodeBERT, now dominate benchmark "
 "leaderboards [8], [9], while CodeSearchNet established neural code retrieval as a task in its "
 "own right [10]. Two gaps motivate the present work. First, almost all of this literature "
 "reports only an aggregate label, so nothing establishes that a model attends to the statements "
 "that constitute the vulnerability; attention maps are routinely displayed as explanations "
 "without being evaluated, a practice Jain and Wallace criticised directly [13]. Second, "
 "detection and retrieval are treated separately, although an analyst triaging a codebase needs "
 "both: a verdict, and a ranked list of comparable code.")
para(doc, "This project therefore builds a single encoder trained on three objectives at once: "
 "detection, attention guidance against statement-level taint annotations, and supervised "
 "contrastive retrieval with the vulnerability family as the relevance key [11]. The attention "
 "formulation follows Bahdanau-style additive attention [12], and the guidance term follows the "
 "rationale-supervision tradition exemplified by ERASER [14]. The intended beneficiaries are "
 "security engineers who must triage scanner output, and code-review tooling that must rank "
 "findings rather than merely emit them.")
para(doc, "The report proceeds as follows. The problem is formalised as a taint-reachability "
 "search and its variables defined; the AGNCS architecture, its three loss terms and the "
 "baseline algorithms are specified with pseudocode; the corpus, hyper-parameters, metrics and "
 "computing environment are described; results are presented for detection, code search and "
 "attention quality on both an in-distribution and a held-out-family test set; the methods are "
 "critically compared; and the work concludes with recommendations. The findings are mixed, and "
 "are reported as such.")

# ======================================================= PROBLEMS SOLVED
heading(doc, "Description of the problems solved", "20")
sub(doc, "Two coupled tasks")
para(doc, "Let a Python function be represented by its token sequence x = (x₁, …, x_L) "
 "with x_t drawn from a vocabulary V. Two problems are solved jointly.")
rich(doc, [("Detection. ", True, False),
 ("Learn fθ : V^L → [0,1] estimating P(y = 1 | x), where y = 1 denotes the presence of "
  "a SQL-injection defect. The emitted decision is ŷ = 1[fθ(x) ≥ τ], with the "
  "threshold τ selected on a validation split.", False, False)])
rich(doc, [("Code search. ", True, False),
 ("Learn gθ : V^L → S^(d−1), mapping a function onto the unit sphere so that "
  "cosine similarity ranks semantically related code first. For a query q and corpus C the "
  "returned ordering is", False, False)])
eq(doc, "rank(q) = argsort_{c ∈ C} − gθ(q)ᵀ gθ(c).")
para(doc, "The two share one encoder. That sharing is deliberate: the representation that "
 "decides whether code is vulnerable is the same one that decides whether two fragments are "
 "alike.")

sub(doc, "Mathematical model of the defect")
para(doc, "A SQL injection exists when a taint path connects an untrusted source to a "
 "query-construction sink without passing through a sanitiser. Writing G = (Vᵢ, E) for the "
 "intra-procedural data-flow graph of the function, S for the set of source nodes, K for sink "
 "nodes and N for sanitiser nodes, the function is vulnerable if and only if")
eq(doc, "∃ v₀ ∈ S, vₙ ∈ K : v₀ → v₁ → ⋯ "
        "→ vₙ ∈ E* ∧ {v₀, …, vₙ} ∩ N = ∅.")
para(doc, "The discriminating detail is the manner in which the value reaches the sink. The two "
 "statements below are nearly identical as text and opposite in meaning:")
code(doc, 'cur.execute(f"SELECT * FROM t WHERE u = \'{name}\'")    # vulnerable')
code(doc, 'cur.execute("SELECT * FROM t WHERE u = ?", (name,))     # safe')
para(doc, "In the first, attacker-controlled text becomes part of the statement; in the second "
 "it is handed to the driver as data and can never alter the parse. Every informative signal in "
 "this problem is therefore a structural relation rather than a word, which is what defeats "
 "lexical rules and motivates a representation that makes interpolation explicit.", before=6)

sub(doc, "Variables and parameters")
table(doc, ["Symbol", "Domain", "Meaning"], [
 ["x", "V^L", "token sequence of one function; L = 160 after truncation"],
 ["y", "{0, 1}", "label; 1 = SQL-injection vulnerable (CWE-89)"],
 ["φ", "{1, …, 30}", "vulnerability/safety family; relevance key for retrieval"],
 ["m", "{0, 1}^L", "statement-level taint or safety-evidence annotation, per token"],
 ["H", "R^(L×d)", "contextual token states from the encoder, d = 96"],
 ["α", "Δ^(L−1)", "attention distribution over tokens (sums to 1)"],
 ["c", "R^d", "pooled code vector"],
 ["z", "S^(d′−1)", "L2-normalised search embedding, d′ = 96"],
 ["λ_a, λ_r", "R₀⁺", "weights of the guidance and contrastive loss terms"],
 ["κ", "R⁺", "contrastive temperature, κ = 0.1"],
 ["τ", "[0, 1]", "decision threshold, selected on validation only"],
], [15, 16, 69])
caption(doc, "Table 1. Variables and parameters of the model.")

sub(doc, "Relevance to artificial intelligence")
para(doc, "Deciding membership of the set defined above is a reachability search over a "
 "data-flow graph, which places the problem squarely in the Searching area of the module. "
 "Classical static analysis performs that search symbolically: it is exact for the patterns its "
 "rules encode and blind to everything else. This project replaces the symbolic search with a "
 "learned one. The encoder embeds code so that functions realising the same taint pattern fall "
 "close together, detection becomes a question about a neighbourhood in that space, and "
 "attention is the mechanism selecting which tokens the search consults. The central hypothesis "
 "is that supervising that attention makes the learned search transfer to patterns absent from "
 "training. The work additionally draws on representation learning, metric learning via a "
 "supervised contrastive objective, and the interpretability question of whether an attention "
 "map constitutes evidence.")
pagebreak(doc)

# ======================================================= ALGORITHMS
heading(doc, "Description of the algorithms implemented", "15")
sub(doc, "AGNCS architecture")
para(doc, "Token identifiers are embedded and summed with a learned positional embedding, then "
 "passed through three pre-normalised residual blocks. Each block applies a depth-wise dilated "
 "convolution (kernel 3, dilations 1, 2 and 4), a point-wise expansion, a gated linear unit "
 "[17] and a projection, giving a receptive field of R = 1 + 2(1+2+4) = 15 tokens [16]. The "
 "token states are pooled by multi-head additive attention, and the pooled vector feeds two "
 "heads: a detection head emitting a logit, and a search head emitting an L2-normalised "
 "embedding. A dilated convolutional encoder was chosen over a Transformer encoder on measured "
 "grounds: at equal width the Transformer ran at 96 samples per second against 376 on the "
 "target CPU, a 3.9× gap that determines whether a seven-method, three-seed study is "
 "feasible at all. The global token interaction self-attention would supply is instead provided "
 "by the attention pooling layer, which is in any case the quantity this project supervises.")

sub(doc, "Attention pooling")
para(doc, "For head h and position t, with s_t the t-th token state:")
eq(doc, "e_{h,t} = v_hᵀ tanh(W_h s_t + b_h),    α_{h,t} = softmax_t(e_{h,t}),")
eq(doc, "c_h = Σ_t α_{h,t} s_t,    c = W_o [c₁; …; c_H],    "
        "α_t = (1/H) Σ_h α_{h,t}.")
para(doc, "Padding positions are masked to −∞ before the softmax. The head-averaged "
 "α_t is both the map displayed as an explanation and the quantity the guidance loss acts "
 "on. Four heads were used.")

sub(doc, "The three loss terms")
rich(doc, [("Detection. ", True, False), ("Binary cross-entropy over the batch:", False, False)])
eq(doc, "L_cls = −(1/B) Σᵢ [ yᵢ log pᵢ + (1−yᵢ) "
        "log(1−pᵢ) ].")
rich(doc, [("Attention guidance. ", True, False),
 ("With q_t = m_t / Σ m_{t′} the normalised annotation, the cross-entropy between "
  "attention and annotation is taken over the samples B⁺ whose annotation is non-empty:",
  False, False)])
eq(doc, "L_att = −(1/|B⁺|) Σ_{i∈B⁺} Σ_t q_{i,t} log α_{i,t} "
        "= H(q) + D_KL(q ‖ α).")
para(doc, "This term has an irreducible floor equal to the annotation's own entropy "
 "H(q) = log|m|, roughly 3.4 nats for a typical 30-token annotation. A raw value near 3.5 "
 "therefore indicates near-perfect agreement rather than a large residual, so the training log "
 "and Figure 4 report D_KL(q ‖ α) = L_att − H(q), which is zero at a perfect "
 "match. The two differ by a parameter-independent constant and have identical gradients.")
rich(doc, [("Contrastive search. ", True, False),
 ("The supervised contrastive loss of Khosla et al. [11], with the vulnerability family rather "
  "than the binary label as the grouping key, and P(i) the in-batch samples sharing i's family:",
  False, False)])
eq(doc, "L_ret = Σᵢ (−1/|P(i)|) Σ_{p∈P(i)} log [ "
        "exp(zᵢᵀz_p/κ) / Σ_{a≠i} exp(zᵢᵀz_a/κ) ].")
para(doc, "Using the family matters: two distinct injection idioms should not collapse onto one "
 "point merely because both are vulnerable. The objective optimised is")
eq(doc, "L = L_cls + λ_a L_att + λ_r L_ret,    λ_a = 0.5,  λ_r = 0.3.")

sub(doc, "Pseudocode")
for line in [
 "ALGORITHM 1  AGNCS -- training",
 "input :  D = {(x_i, y_i, m_i, phi_i)}     annotated training corpus",
 "         lambda_a, lambda_r              loss weights",
 "         E, B, eta                       epochs, batch size, base LR",
 "output:  theta*                          parameters of the best epoch",
 " 1  theta <- init() ;  best <- -inf ;  stale <- 0",
 " 2  for epoch = 1 .. E do",
 " 3      for each batch Bt ~ FAMILY-BALANCED-SAMPLER(D, B) do",
 " 4          eta_t  <- WARMUP-COSINE(step, eta)",
 " 5          H, pad <- ENCODER(x)",
 " 6          alpha  <- SOFTMAX(SCORE(H) masked by pad)",
 " 7          c      <- sum_t alpha_t * H_t",
 " 8          p      <- sigmoid(CLS-HEAD(c))",
 " 9          z      <- NORMALISE(SEARCH-HEAD(c))",
 "10          L      <- BCE(p, y)",
 "11          if lambda_a > 0 then",
 "12              q <- m / sum(m)                 // rows with sum(m) > 0",
 "13              L <- L + lambda_a * ( - mean_i sum_t q_it log alpha_it )",
 "14          if lambda_r > 0 then",
 "15              L <- L + lambda_r * SUPCON(z, phi, kappa)",
 "16          theta <- ADAMW(theta, CLIP(grad L, 1.0), eta_t)",
 "17      f1 <- EVALUATE(theta, VAL)",
 "18      if f1 > best then best <- f1 ; theta* <- theta ; stale <- 0",
 "19      else stale <- stale + 1 ; if stale >= patience then break",
 "20  return theta*",
]:
    code(doc, line)
para(doc, "", after=4)
for line in [
 "ALGORITHM 2  FAMILY-BALANCED-SAMPLER",
 "input :  D, batch size B, families per batch F",
 " 1  repeat |D| / B times",
 " 2      S <- sample F distinct families without replacement",
 " 3      batch <- {}",
 " 4      for each family f in S do",
 " 5          batch <- batch  U  sample( floor(B/F) items of family f )",
 " 6      yield batch",
 "-- rationale: L_ret needs several members of a family inside one batch,",
 "   otherwise P(i) is empty and the contrastive term is degenerate.",
]:
    code(doc, line)
para(doc, "", after=4)
for line in [
 "ALGORITHM 3  Neural code search (inference)",
 "input :  query function q, corpus C, trained theta*",
 " 1  z_q <- NORMALISE(SEARCH-HEAD(POOL(ENCODER(q))))",
 " 2  for each c in C do  z_c <- cached embedding of c",
 " 3  s_c <- z_q . z_c                           // cosine similarity",
 " 4  return C sorted by s_c descending",
 "-- the corpus is embedded once; a query costs one forward pass and one matmul.",
]:
    code(doc, line)
para(doc, "", after=6)

sub(doc, "Baseline algorithms")
rich(doc, [("M1, pattern-matching analyser. ", True, False),
 ("A reimplementation of the published rule family behind Bandit B608 and semgrep's "
  "formatted-SQL-query rules [3], [4]. The abstract syntax tree is walked; for every call whose "
  "attribute is an execute-like sink, the first argument is resolved to its definitions, and the "
  "rule fires if that expression is produced by f-string interpolation, %-formatting, str.format "
  "or + concatenation and contains SQL keywords.", False, False)])
rich(doc, [("M2, AST taint analyser. ", True, False),
 ("A genuine intra-procedural forward taint analysis. Function parameters and request objects "
  "seed the taint set; assignments, string operations, for-loop and comprehension targets and "
  "container accumulation propagate it to a fixed point; int() and float() are recognised "
  "sanitisers; execute-like calls, including keyword forms such as .extra(where=…), are "
  "sinks. Because it resolves variables to their definitions and follows a local helper's return "
  "value, it detects multi-hop and inter-procedural flows that M1 cannot see. It is deliberately "
  "not a strawman: the comparison is only informative against a static analyser that is actually "
  "good.", False, False)])
rich(doc, [("M3, TF-IDF with logistic regression. ", True, False),
 ("Word unigrams and bigrams over the same token stream the neural models receive, sublinear "
  "term frequency, minimum document frequency 3, and L2-regularised logistic regression with "
  "C = 4. Operating on identical preprocessing isolates the contribution of the architecture "
  "from that of the features. Its L2-normalised vectors additionally serve as a classical "
  "retrieval index.", False, False)])

sub(doc, "Ablation design")
para(doc, "Each variant removes exactly one component, so every row of the results table "
 "isolates one claim.")
table(doc, ["Variant", "Pooling", "λ_a", "λ_r", "Isolates"], [
 ["M4", "mean", "0", "0", "the value of attention at all"],
 ["M5", "attention", "0", "0", "the value of guiding the attention"],
 ["M6", "attention", "0", "0.3", "the value of the contrastive search head"],
 ["M7 (proposed)", "attention", "0.5", "0.3", "the full method"],
], [16, 14, 7, 7, 56])
caption(doc, "Table 2. Ablation design. All four variants share the same encoder and budget.")
pagebreak(doc)

import report_sections2
report_sections2.write(doc)

doc.save(OUT)
import os
print("wrote %s  (%.0f KB, %d paragraphs, %d tables)"
      % (os.path.basename(OUT), os.path.getsize(OUT) / 1024,
         len(doc.paragraphs), len(doc.tables)))
