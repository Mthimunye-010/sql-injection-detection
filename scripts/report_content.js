// NCOS734 PG4 mini-project report -- content.
//   node scripts/report_content.js   ->  NCOS734_PG4_Report.docx

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber,
  PageBreak, LevelFormat, TabStopType, BorderStyle,
} = require("docx");
const H = require("./build_report.js");
const { P, RP, H1, H2, BULLET, NUM, CODE, EQ, CAPTION, table, figure,
        FONT, SZ, SZ_SMALL, CONTENT_W, ROOT } = H;

const children = [];
const add = (...xs) => xs.forEach((x) => children.push(x));

// =========================================================== TITLE PAGE
const TITLE = "Attention-Guided Neural Code Search for Automated SQL Injection "
            + "Vulnerability Detection in Python Source Code";

add(new Paragraph({ spacing: { before: 1200, after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "NCOS734 Semester Project", font: FONT, size: 32, bold: true })] }));
add(new Paragraph({ spacing: { after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: TITLE, font: FONT, size: 28, bold: true })] }));
add(new Paragraph({ spacing: { after: 700 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Session 2026", font: FONT, size: 24 })] }));

[1, 2, 3, 4, 5].forEach((n) => add(new Paragraph({
  spacing: { after: 100 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: `[ Insert ${n === 1 ? "1st" : n === 2 ? "2nd" : n === 3 ? "3rd" : n + "th"} `
        + "Group Member name (Student Number) here ]", font: FONT, size: SZ })] })));

add(new Paragraph({ spacing: { before: 600, after: 100 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "[ Insert Department, Faculty and University here ]",
                           font: FONT, size: SZ })] }));
add(new Paragraph({ spacing: { before: 400, after: 100 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Module Lecturer: MA Arasomwan", font: FONT, size: SZ })] }));
add(new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "October 05, 2026", font: FONT, size: SZ })] }));
add(new Paragraph({ children: [new PageBreak()] }));

// =========================================================== ABSTRACT
add(H1("Abstract", "12"));
add(RP([["Background: ", { bold: true }], ["SQL injection (CWE-89) remains among the most "
 + "damaging classes of web-application vulnerability. Static analysers detect it by matching "
 + "syntactic patterns and are widely deployed in continuous-integration pipelines, but their "
 + "output is dominated by false positives: safe code that merely resembles a dangerous "
 + "construct — an integer cast, an allow-listed column name, a regex-validated token — "
 + "trips the same rules. The review effort spent discarding spurious alerts, rather than any "
 + "shortfall in recall, is the dominant cost of such tools."]]));
add(RP([["Aim: ", { bold: true }], ["This project tests whether supervising a neural model’s "
 + "attention mechanism — training it to attend to the statements that actually carry the "
 + "taint flow — lowers the false-positive rate relative to static analysis, and whether the "
 + "resulting representation supports neural code search over vulnerability patterns."]]));
add(RP([["Method: ", { bold: true }], ["A corpus of 60,000 unique, syntactically valid Python "
 + "functions was generated across 30 vulnerability and safety families, each annotated at "
 + "statement level with its taint path or its safety evidence, with six families withheld from "
 + "training entirely to measure generalisation. AGNCS, a dilated convolutional encoder with "
 + "multi-head additive attention pooling, was trained jointly on detection (binary "
 + "cross-entropy), attention guidance (cross-entropy to the normalised annotation) and "
 + "supervised contrastive retrieval. Seven methods were compared over three seeds: two static "
 + "analysers, a TF-IDF classifier, and four neural ablations."]]));
add(RP([["Findings: ", { bold: true }], ["The premise was confirmed. The AST taint analyser "
 + "achieved recall 1.000 on both test splits while carrying a false-positive rate of 0.167 "
 + "in-distribution and 0.667 on held-out safe code, concentrated entirely on hard negatives. "
 + "The proposed remedy was only partly supported. On held-out families TF-IDF with logistic "
 + "regression attained F1 0.829 against AGNCS’s 0.600, and an attention-free control "
 + "generalised better than any attention variant. Attention guidance nevertheless produced two "
 + "statistically significant effects: held-out false positives fell from 0.908 to 0.325 "
 + "(p = 0.011, d₂ = −5.42) and the overlap between the attention map and the annotated "
 + "taint path rose from 0.506 to 0.971. Code search reached 0.912 mean average precision on "
 + "held-out families against 0.796 for the strongest baseline."]]));
add(RP([["Conclusion: ", { bold: true }], ["Attention-guided neural code search succeeds as a "
 + "retrieval and ranking mechanism but does not outperform a lexical baseline at detection on "
 + "unseen injection patterns. The deployable configuration is therefore a pipeline rather than "
 + "a replacement: taint analysis supplies recall, and the guided embedding supplies the "
 + "ordering that decides what a human reviews first."]]));

// =========================================================== INTRODUCTION
add(H1("Introduction", "12"));
add(P("SQL injection arises when untrusted input is concatenated into a database query instead "
 + "of being bound as a parameter. Catalogued as CWE-89 and still present in the OWASP Top Ten "
 + "under Injection [1], [2], it is conceptually well understood and nevertheless persistent, "
 + "because the defective and the correct idiom differ by only a few characters of source text."));
add(P("The tools deployed against it are predominantly static. Rule-based scanners such as "
 + "Bandit’s B608 check and the corresponding semgrep rule family flag any query argument "
 + "constructed by string formatting or concatenation [3], [4]. These rules are cheap and "
 + "transparent, but they reason about surface form rather than data flow, so they cannot "
 + "distinguish interpolation of an attacker-controlled string from interpolation of an integer "
 + "cast or of a column name checked against an allow-list. The consequence is an alert queue "
 + "that developers learn to ignore — a failure of precision, not of recall."));
add(P("Learning-based detection has been pursued as an alternative. VulDeePecker introduced "
 + "recurrent models over sliced code gadgets [5]; Devign applied graph neural networks to code "
 + "property graphs [6]; Big-Vul supplied large-scale mined labels [7]. Transformer models "
 + "pre-trained on source code, notably CodeBERT and GraphCodeBERT, now dominate benchmark "
 + "leaderboards [8], [9], while CodeSearchNet established neural code retrieval as a task in "
 + "its own right [10]. Two gaps motivate the present work. First, almost all of this literature "
 + "reports only an aggregate label, so nothing establishes that a model attends to the "
 + "statements that constitute the vulnerability; attention maps are routinely displayed as "
 + "explanations without being evaluated, a practice Jain and Wallace criticised directly [13]. "
 + "Second, detection and retrieval are treated separately, although an analyst triaging a "
 + "codebase needs both: a verdict and a ranked list of comparable code."));
add(P("This project therefore builds a single encoder trained on three objectives at once: "
 + "detection, attention guidance against statement-level taint annotations, and supervised "
 + "contrastive retrieval with the vulnerability family as the relevance key [11]. The attention "
 + "formulation follows Bahdanau-style additive attention [12], and the guidance term follows "
 + "the rationale-supervision tradition exemplified by ERASER [14]. The intended beneficiaries "
 + "are security engineers who must triage scanner output, and code-review tooling that must "
 + "rank findings rather than merely emit them."));
add(P("The report proceeds as follows. The problem is formalised as a taint-reachability search "
 + "and its variables defined; the AGNCS architecture, its three loss terms and the baseline "
 + "algorithms are specified with pseudocode; the corpus, hyper-parameters, metrics and "
 + "computing environment are described; results are presented for detection, code search and "
 + "attention quality on both an in-distribution and a held-out-family test set; the methods are "
 + "critically compared; and the work concludes with recommendations. The findings are mixed, and "
 + "are reported as such."));

// =========================================================== PROBLEMS SOLVED
add(H1("Description of the problems solved", "20"));
add(H2("Two coupled tasks"));
add(P("Let a Python function be represented by its token sequence x = (x₁, …, x_L) with "
 + "x_t drawn from a vocabulary V. Two problems are solved jointly."));
add(RP([["Detection. ", { bold: true }], ["Learn fθ : V^L → [0,1] estimating "
 + "P(y = 1 | x), where y = 1 denotes the presence of a SQL-injection defect. The emitted "
 + "decision is ŷ = 1[fθ(x) ≥ τ], with the threshold τ selected on a "
 + "validation split."]]));
add(RP([["Code search. ", { bold: true }], ["Learn gθ : V^L → S^(d−1), mapping a "
 + "function onto the unit sphere so that cosine similarity ranks semantically related code "
 + "first. For a query q and corpus C the returned ordering is"]]));
add(EQ("rank(q) = argsort_{c ∈ C} − gθ(q)ᵀ gθ(c)."));
add(P("The two share one encoder. That sharing is deliberate: the representation that decides "
 + "whether code is vulnerable is the same one that decides whether two fragments are alike."));

add(H2("Mathematical model of the defect"));
add(P("A SQL injection exists when a taint path connects an untrusted source to a "
 + "query-construction sink without passing through a sanitiser. Writing G = (Vᵢ, E) for the "
 + "intra-procedural data-flow graph of the function, S for the set of source nodes, K for sink "
 + "nodes and N for sanitiser nodes, the function is vulnerable if and only if"));
add(EQ("∃ v₀ ∈ S, vₙ ∈ K : v₀ → v₁ → ⋯ → vₙ "
 + "∈ E* ∧ {v₀, …, vₙ} ∩ N = ∅."));
add(P("The discriminating detail is the manner in which the value reaches the sink. The two "
 + "statements below are nearly identical as text and opposite in meaning:"));
add(CODE('cur.execute(f"SELECT * FROM t WHERE u = \'{name}\'")    # vulnerable'));
add(CODE('cur.execute("SELECT * FROM t WHERE u = ?", (name,))     # safe'));
add(P("In the first, attacker-controlled text becomes part of the statement; in the second it is "
 + "handed to the driver as data and can never alter the parse. Every informative signal in this "
 + "problem is therefore a structural relation rather than a word, which is what defeats "
 + "lexical rules and motivates a representation that makes interpolation explicit.", { after: 160 }));

add(H2("Variables and parameters"));
add(table(
  ["Symbol", "Domain", "Meaning"],
  [
    ["x", "V^L", "token sequence of one function; L = 160 after truncation"],
    ["y", "{0, 1}", "label; 1 = SQL-injection vulnerable (CWE-89)"],
    ["φ", "{1, …, 30}", "vulnerability/safety family; the relevance key for retrieval"],
    ["m", "{0, 1}^L", "statement-level taint or safety-evidence annotation, per token"],
    ["H", "R^(L×d)", "contextual token states produced by the encoder, d = 96"],
    ["α", "Δ^(L−1)", "attention distribution over tokens (sums to 1)"],
    ["c", "R^d", "pooled code vector"],
    ["z", "S^(d′−1)", "L2-normalised search embedding, d′ = 96"],
    ["λ_a, λ_r", "R₀⁺", "weights of the guidance and contrastive loss terms"],
    ["κ", "R⁺", "contrastive temperature, κ = 0.1"],
    ["τ", "[0, 1]", "decision threshold, selected on validation only"],
  ], [1400, 1500, 6700]));
add(CAPTION("Table 1. Variables and parameters of the model."));

add(H2("Relevance to artificial intelligence"));
add(P("Deciding membership of the set defined above is a reachability search over a data-flow "
 + "graph, which places the problem squarely in the Searching area of the module. Classical "
 + "static analysis performs that search symbolically: it is exact for the patterns its rules "
 + "encode and blind to everything else. This project replaces the symbolic search with a "
 + "learned one. The encoder embeds code so that functions realising the same taint pattern fall "
 + "close together, detection becomes a question about a neighbourhood in that space, and "
 + "attention is the mechanism selecting which tokens the search consults. The central hypothesis "
 + "is that supervising that attention makes the learned search transfer to patterns absent from "
 + "training. The work additionally draws on representation learning, metric learning via a "
 + "supervised contrastive objective, and the interpretability question of whether an attention "
 + "map constitutes evidence."));
add(new Paragraph({ children: [new PageBreak()] }));

// =========================================================== ALGORITHMS
add(H1("Description of the algorithms implemented", "15"));
add(H2("AGNCS architecture"));
add(P("Token identifiers are embedded and summed with a learned positional embedding, then "
 + "passed through three pre-normalised residual blocks. Each block applies a depth-wise "
 + "dilated convolution (kernel 3, dilations 1, 2 and 4), a point-wise expansion, a gated linear "
 + "unit and a projection, giving a receptive field of R = 1 + 2(1+2+4) = 15 tokens. The token "
 + "states are pooled by multi-head additive attention, and the pooled vector feeds two heads: a "
 + "detection head emitting a logit, and a search head emitting an L2-normalised embedding. A "
 + "dilated convolutional encoder was chosen over a Transformer encoder on measured grounds: at "
 + "equal width the Transformer ran at 96 samples per second against 376 on the target CPU, a "
 + "3.9× gap that determines whether a seven-method, three-seed study is feasible at all. "
 + "The global token interaction self-attention would supply is instead provided by the attention "
 + "pooling layer, which is in any case the quantity this project supervises."));

add(H2("Attention pooling"));
add(P("For head h and position t, with s_t the t-th token state:"));
add(EQ("e_{h,t} = v_hᵀ tanh(W_h s_t + b_h),   α_{h,t} = softmax_t(e_{h,t}),"));
add(EQ("c_h = Σ_t α_{h,t} s_t,   c = W_o [c₁; …; c_H],   "
 + "α_t = (1/H) Σ_h α_{h,t}."));
add(P("Padding positions are masked to −∞ before the softmax. The head-averaged "
 + "α_t is both the map displayed as an explanation and the quantity the guidance loss acts "
 + "on. Four heads were used."));

add(H2("The three loss terms"));
add(RP([["Detection. ", { bold: true }], ["Binary cross-entropy over the batch:"]]));
add(EQ("L_cls = −(1/B) Σᵢ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]."));
add(RP([["Attention guidance. ", { bold: true }], ["With q_t = m_t / Σ_{t′} m_{t′} "
 + "the normalised annotation, the cross-entropy between attention and annotation is taken over "
 + "the samples B⁺ whose annotation is non-empty:"]]));
add(EQ("L_att = −(1/|B⁺|) Σ_{i∈B⁺} Σ_t q_{i,t} log α_{i,t} "
 + "= H(q) + D_KL(q ‖ α)."));
add(P("This term has an irreducible floor equal to the annotation’s own entropy "
 + "H(q) = log|m|, roughly 3.4 nats for a typical 30-token annotation. A raw value near 3.5 "
 + "therefore indicates near-perfect agreement rather than a large residual, so the training log "
 + "and Figure 4 report D_KL(q ‖ α) = L_att − H(q), which is zero at a perfect "
 + "match. The two differ by a parameter-independent constant and have identical gradients."));
add(RP([["Contrastive search. ", { bold: true }], ["The supervised contrastive loss of Khosla "
 + "et al. [11], with the vulnerability family rather than the binary label as the grouping key, "
 + "and P(i) the in-batch samples sharing i’s family:"]]));
add(EQ("L_ret = Σᵢ (−1/|P(i)|) Σ_{p∈P(i)} log "
 + "[ exp(zᵢᵀz_p/κ) / Σ_{a≠i} exp(zᵢᵀz_a/κ) ]."));
add(P("Using the family matters: two distinct injection idioms should not collapse onto one "
 + "point merely because both are vulnerable. The objective optimised is"));
add(EQ("L = L_cls + λ_a L_att + λ_r L_ret,   λ_a = 0.5, λ_r = 0.3."));

add(H2("Pseudocode"));
add(CODE("ALGORITHM 1  AGNCS -- training"));
add(CODE("input :  D = {(x_i, y_i, m_i, phi_i)}      annotated training corpus"));
add(CODE("         lambda_a, lambda_r               loss weights"));
add(CODE("         E, B, eta                        epochs, batch size, base LR"));
add(CODE("output:  theta*                           parameters of the best epoch"));
add(CODE(" 1  theta <- init() ;  best <- -inf ;  stale <- 0"));
add(CODE(" 2  for epoch = 1 .. E do"));
add(CODE(" 3      for each batch Bt ~ FAMILY-BALANCED-SAMPLER(D, B) do"));
add(CODE(" 4          eta_t  <- WARMUP-COSINE(step, eta)"));
add(CODE(" 5          H, pad <- ENCODER(x)"));
add(CODE(" 6          alpha  <- SOFTMAX(SCORE(H) masked by pad)"));
add(CODE(" 7          c      <- sum_t alpha_t * H_t"));
add(CODE(" 8          p      <- sigmoid(CLS-HEAD(c))"));
add(CODE(" 9          z      <- NORMALISE(SEARCH-HEAD(c))"));
add(CODE("10          L      <- BCE(p, y)"));
add(CODE("11          if lambda_a > 0 then"));
add(CODE("12              q <- m / sum(m)                  // rows with sum(m) > 0"));
add(CODE("13              L <- L + lambda_a * ( - mean_i sum_t q_it log alpha_it )"));
add(CODE("14          if lambda_r > 0 then"));
add(CODE("15              L <- L + lambda_r * SUPCON(z, phi, kappa)"));
add(CODE("16          theta <- ADAMW(theta, CLIP(grad L, 1.0), eta_t)"));
add(CODE("17      f1 <- EVALUATE(theta, VAL)"));
add(CODE("18      if f1 > best then best <- f1 ; theta* <- theta ; stale <- 0"));
add(CODE("19      else stale <- stale + 1 ; if stale >= patience then break"));
add(CODE("20  return theta*"));
add(P("", { after: 100 }));
add(CODE("ALGORITHM 2  FAMILY-BALANCED-SAMPLER"));
add(CODE("input :  D, batch size B, families per batch F"));
add(CODE(" 1  repeat |D| / B times"));
add(CODE(" 2      S <- sample F distinct families without replacement"));
add(CODE(" 3      batch <- {}"));
add(CODE(" 4      for each family f in S do"));
add(CODE(" 5          batch <- batch  U  sample( floor(B/F) items of family f )"));
add(CODE(" 6      yield batch"));
add(CODE("-- rationale: L_ret needs several members of a family inside one batch,"));
add(CODE("   otherwise P(i) is empty and the contrastive term is degenerate."));
add(P("", { after: 100 }));
add(CODE("ALGORITHM 3  Neural code search (inference)"));
add(CODE("input :  query function q, corpus C, trained theta*"));
add(CODE(" 1  z_q <- NORMALISE(SEARCH-HEAD(POOL(ENCODER(q))))"));
add(CODE(" 2  for each c in C do  z_c <- cached embedding of c"));
add(CODE(" 3  s_c <- z_q . z_c                            // cosine similarity"));
add(CODE(" 4  return C sorted by s_c descending"));
add(CODE("-- the corpus is embedded once; a query costs one forward pass and one matmul."));
add(P("", { after: 160 }));

add(H2("Baseline algorithms"));
add(RP([["M1, pattern-matching analyser. ", { bold: true }], ["A reimplementation of the "
 + "published rule family behind Bandit B608 and semgrep’s formatted-SQL-query rules [3], "
 + "[4]. The abstract syntax tree is walked; for every call whose attribute is an execute-like "
 + "sink, the first argument is resolved to its definitions, and the rule fires if that "
 + "expression is produced by f-string interpolation, %-formatting, str.format or + "
 + "concatenation and contains SQL keywords."]]));
add(RP([["M2, AST taint analyser. ", { bold: true }], ["A genuine intra-procedural forward "
 + "taint analysis. Function parameters and request objects seed the taint set; assignments, "
 + "string operations, for-loop and comprehension targets and container accumulation propagate "
 + "it to a fixed point; int() and float() are recognised sanitisers; execute-like calls, "
 + "including keyword forms such as .extra(where=…), are sinks. Because it resolves "
 + "variables to their definitions and follows a local helper’s return value, it detects "
 + "multi-hop and inter-procedural flows that M1 cannot see. It is deliberately not a strawman: "
 + "the comparison is only informative against a static analyser that is actually good."]]));
add(RP([["M3, TF-IDF with logistic regression. ", { bold: true }], ["Word unigrams and bigrams "
 + "over the same token stream the neural models receive, sublinear term frequency, minimum "
 + "document frequency 3, L2-regularised logistic regression with C = 4. Operating on identical "
 + "preprocessing isolates the contribution of the architecture from that of the features. Its "
 + "L2-normalised vectors additionally serve as a classical retrieval index."]]));

add(H2("Ablation design"));
add(P("Each variant removes exactly one component, so every row of the results table isolates "
 + "one claim."));
add(table(["Variant", "Pooling", "λ_a", "λ_r", "Isolates"],
  [["M4", "mean", "0", "0", "the value of attention at all"],
   ["M5", "attention", "0", "0", "the value of guiding the attention"],
   ["M6", "attention", "0", "0.3", "the value of the contrastive search head"],
   ["M7 (proposed)", "attention", "0.5", "0.3", "the full method"]],
  [1500, 1300, 700, 700, 5400]));
add(CAPTION("Table 2. Ablation design. All four variants share the same encoder and budget."));
add(new Paragraph({ children: [new PageBreak()] }));

// =========================================================== MATERIALS & METHODS
add(H1("Materials and methods", "20"));
add(H2("Dataset: acquisition and description"));
add(P("The datasets named in the project brief were evaluated and rejected for concrete "
 + "reasons. Juliet and SARD are overwhelmingly C, C++ and Java, and their Python content does "
 + "not cover modern ORM and f-string idioms. Devign and Big-Vul are C and C++ only and carry "
 + "function-level binary labels, with no statement-level taint annotation [6], [7]. "
 + "CodeSearchNet is Python and large but contains no vulnerability labels at all [10]. This "
 + "experiment requires three annotations simultaneously — a binary label, a "
 + "statement-level taint annotation to supervise attention, and a family label to define "
 + "retrieval relevance — and no public corpus supplies the combination for Python."));
add(P("The corpus was therefore synthesised from a taint-aware grammar, which makes the "
 + "annotation exact by construction and free of the label noise that affects mined "
 + "vulnerability datasets. This is the principal limitation of the work and is treated as such "
 + "in the critical evaluation. Fifteen vulnerable families span direct f-string interpolation, "
 + "%-formatting, str.format, concatenation, multi-hop aliasing, inter-procedural helpers, "
 + "request-dictionary sources, loop-built WHERE clauses, ORM raw-SQL escape hatches, "
 + "multi-statement execution, string.Template substitution, identifier injection in ORDER BY, "
 + "fragment joining, partial parameterisation and ineffective sanitisers. Fifteen safe families "
 + "cover the correct idioms, and eight of them are deliberate hard negatives: safe code that "
 + "still interpolates a value into a SQL string via an integer cast, an allow-listed "
 + "identifier, an enum fragment lookup, a regex-validated token, module constants, a "
 + "composable-SQL API, SQL text that is never executed, or a non-SQL sink."));
add(P("Controls keep the task honest. Vulnerable and safe samples draw identifiers, tables, "
 + "columns, docstrings, decorators and comments from the same pools, so no name is predictive. "
 + "Comments are adversarial, with “input is already sanitised” appearing on both "
 + "classes. Five database APIs, seven wrapping styles, optional type hints and zero to three "
 + "distractor statements per sample are all drawn independently of the label. Every sample is "
 + "validated with ast.parse, giving a 100% syntactic validity rate, and the corpus is "
 + "de-duplicated on the SHA-1 of its source."));
add(table(["Property", "Value"],
  [["Total functions / unique source", "60,000 / 60,000"],
   ["Families (vulnerable / safe)", "30 (15 / 15)"],
   ["Hard negatives", "14,834"],
   ["Class balance (safe / vulnerable)", "30,003 / 29,997"],
   ["Lines per function (mean / median / range)", "11.5 / 11 / 2–27"],
   ["Annotated statements per function (mean)", "2.43"],
   ["Database APIs", "sqlite3, psycopg2, mysql.connector, pymysql, pyodbc"],
   ["Difficulty (easy / medium / hard)", "17,336 / 17,334 / 25,330"],
   ["Syntactic validity (ast.parse)", "100%"]],
  [4600, 5000]));
add(CAPTION("Table 3. Corpus summary."));
add(P("Twenty-four families form the in-distribution pool, split 70/10/20 into train (36,402), "
 + "validation (5,208) and test-ID (10,392). Six families — three vulnerable "
 + "(inter-procedural helper, string.Template, fragment joining) and three safe (enum fragment "
 + "lookup, never-executed SQL, composable-SQL API) — are withheld from training entirely "
 + "and constitute test-OOD (7,998). All six are rated hard, and all three held-out safe "
 + "families are hard negatives, so the held-out false-positive rate is a pure hard-negative "
 + "measure and is not directly comparable with the in-distribution figure. Every split is "
 + "balanced to a positive rate of 0.500."));
add(P("A validity check was run on the risk that identifier anonymisation collapses distinct "
 + "functions onto identical token sequences. Measured on the encoded tensors, 99.93% of "
 + "training sequences are unique, and only 0.14% of test-ID and 0.00% of test-OOD sequences "
 + "coincide with a training sequence, so the test splits are not measuring memorisation."));

add(H2("Tokenisation"));
add(P("Three decisions carry most of the inductive bias. Identifiers outside a fixed API "
 + "lexicon are rewritten to <ID_k>, with k the order of first appearance within that sample; "
 + "this destroys name memorisation while preserving aliasing, so a variable written on one line "
 + "and read on another is the same symbol in both places. String literals are decomposed into "
 + "structural markers, so that a quoted interpolation (<SQ> <FFIELD_BEG> <ID_k> <FFIELD_END> "
 + "<SQ>), a question-mark placeholder (<QMARK>) and a %s placeholder (<PCT_FMT>) are distinct "
 + "tokens; without this the vulnerable and safe idioms are nearly indistinguishable. A "
 + "hand-written scanner is used rather than the standard tokenize module, because CPython 3.12 "
 + "changed f-string tokenisation and the notebook must reproduce on another interpreter. The "
 + "resulting vocabulary is 415 types."));

add(H2("Experimental setup and parameter settings"));
add(table(["Parameter", "Value", "Parameter", "Value"],
  [["Sequence length L", "160 (head+tail trunc.)", "Optimiser", "AdamW"],
   ["Encoder width d", "96", "Base learning rate", "2.5 × 10⁻³"],
   ["Dilations / receptive field", "1, 2, 4 / 15 tokens", "Schedule", "warm-up 10% + cosine"],
   ["Attention heads H", "4", "Weight decay", "1 × 10⁻⁴"],
   ["Search embedding d′", "96", "Gradient clipping", "1.0"],
   ["Dropout", "0.1", "Batch size", "128"],
   ["λ_a / λ_r / κ", "0.5 / 0.3 / 0.1", "Families per batch", "16"],
   ["Trainable parameters", "212,385", "Epochs / patience", "5 / 2"],
   ["Training subset", "12,000 (stratified)", "Seeds", "0, 1, 2"]],
  [2500, 2400, 2400, 2300]));
add(CAPTION("Table 4. Hyper-parameters. The receptive field was selected as described below."));
add(P("The threshold τ and the early-stopping epoch are both chosen on the validation "
 + "split; test-ID and test-OOD are read once, at the end. The receptive field was chosen by a "
 + "dedicated ablation. Measuring the token distance between the first and last annotated "
 + "statement gives a median span of 37 tokens in-distribution and 58 on held-out families, which "
 + "predicts that a wide field is necessary. Testing R ∈ {15, 31, 63, 127} refuted that "
 + "prediction: in-distribution F1 varied by only 0.0013 while held-out F1 varied by 0.121, and a "
 + "wider field correlated with a worse held-out false-positive rate (Spearman ρ = +0.80) "
 + "rather than a better one. Because in-distribution validation F1 is saturated near 0.999 for "
 + "every configuration, no legitimate selection signal distinguishes them; choosing R by its "
 + "held-out score would constitute selection on the test set. R was therefore selected on "
 + "validation, where all four tie, with computational cost as the tie-breaker, giving dilations "
 + "1, 2, 4. The ablation is reported as inconclusive about which field is better."));

add(H2("Performance evaluation criteria"));
add(P("Writing TP, FP, TN and FN for the confusion-matrix entries at threshold τ and N for "
 + "the number of evaluated samples:"));
add(EQ("Accuracy = (TP+TN)/N,   Precision = TP/(TP+FP),   Recall = TP/(TP+FN),"));
add(EQ("F₁ = 2·Precision·Recall/(Precision+Recall),   FPR = FP/(FP+TN),"));
add(EQ("MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))."));
add(P("Recall is what a security team values first, because a missed defect ships. The "
 + "false-positive rate is the headline metric of this study, because the project’s premise "
 + "is that static analysers are too noisy to use rather than insufficiently sensitive: each "
 + "spurious alert consumes review budget. MCC is reported because it remains informative under "
 + "imbalance, taking +1 for perfect agreement, 0 for chance and −1 for total disagreement. "
 + "ROC-AUC and PR-AUC summarise ranking across thresholds and are undefined for the rule-based "
 + "analysers, which emit a hard decision; those cells are marked —."));
add(P("For code search, a corpus item is relevant to a query when it belongs to the same "
 + "vulnerability family, that is, realises the same taint pattern. With |Q| queries:"));
add(EQ("MRR = (1/|Q|) Σ_q 1/rank_q,   Recall@k = (1/|Q|) Σ_q 1[relevant item in top k],"));
add(EQ("MAP = (1/|Q|) Σ_q (1/R_q) Σ_{j: rel} P@j,   "
 + "nDCG@k = Σ_{j≤1} rel_j/log₂(j+1) ÷ IDCG@k."));
add(P("MRR answers how far an analyst must scroll before the first useful hit; MAP and nDCG "
 + "reward getting the whole ordering right, nDCG with an explicit position discount. Queries "
 + "are evaluated leave-one-out against a stratified corpus of 6,000 items."));
add(P("Attention quality is evaluated directly, which most vulnerability-detection work omits. "
 + "With valid_i the non-padding positions, k_i = |m_i| and T_i the k_i most-attended tokens:"));
add(EQ("AMR_i = Σ_t α_{i,t} m_{i,t},   "
 + "Lift_i = AMR_i ÷ (|m_i| / |valid_i|),"));
add(EQ("Pointing_i = 1[argmax_t α_{i,t} ∈ m_i],   "
 + "IoU_i = |T_i ∩ m_i| / |T_i ∪ m_i|."));
add(P("Lift divides out the mass a uniform map would place on the annotation, so a lift of 1 "
 + "means no better than attending everywhere, and the figure is comparable across functions of "
 + "differing length. Each neural variant is trained with three seeds; tables report mean "
 + "± standard deviation, and improvement claims are supported by a paired t-test and a "
 + "Wilcoxon signed-rank test with Cohen’s d₂ for effect size."));

add(H2("Software and hardware"));
add(P("All experiments were run on an AMD Ryzen (Family 25) CPU with 12 logical cores, six "
 + "PyTorch threads, under Windows 11 Pro (build 26200), using Python 3.12.0, PyTorch 2.14.1 "
 + "(CPU build), NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1 and Matplotlib "
 + "3.11.2. No GPU was used. Corpus generation takes approximately 11 seconds; tokenising and "
 + "encoding all 60,000 samples takes about 32 seconds and is cached. The full comparative study "
 + "— twelve neural training runs plus baselines, tables and figures — completed in "
 + "56.2 minutes, with individual neural runs averaging 238 to 279 seconds. The static analysers "
 + "require no training and run at 0.227 ms (M1) and 0.794 ms (M2) per function."));
add(new Paragraph({ children: [new PageBreak()] }));

module.exports = { children, add };
