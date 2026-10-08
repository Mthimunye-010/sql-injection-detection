"""Report sections: materials and methods, results, evaluation, conclusion,
references.  Called by write_report.py with the live Document."""

from __future__ import annotations

from build_report import (EM, EN, bullet, caption, code, eq, figure, heading,
                          pagebreak, para, rich, sub, table)


def write(doc) -> None:
    _materials(doc)
    _results(doc)
    _critical(doc)
    _conclusion(doc)
    _references(doc)


# --------------------------------------------------------------------------- #
def _materials(doc) -> None:
    heading(doc, "Materials and methods", "20")

    sub(doc, "Dataset: acquisition and description")
    para(doc, "The datasets named in the project brief were evaluated and rejected for concrete "
         "reasons. Juliet and SARD are overwhelmingly C, C++ and Java, and their Python content "
         "does not cover modern ORM and f-string idioms [15]. Devign and Big-Vul are C and C++ "
         "only and carry function-level binary labels, with no statement-level taint annotation "
         "[6], [7]. CodeSearchNet is Python and large but contains no vulnerability labels at "
         "all [10]. This experiment requires three annotations simultaneously " + EM + " a binary "
         "label, a statement-level taint annotation to supervise attention, and a family label "
         "to define retrieval relevance " + EM + " and no public corpus supplies the combination "
         "for Python.")
    para(doc, "The corpus was therefore synthesised from a taint-aware grammar, which makes the "
         "annotation exact by construction and free of the label noise that affects mined "
         "vulnerability datasets. This is the principal limitation of the work and is treated as "
         "such in the critical evaluation. Fifteen vulnerable families span direct f-string "
         "interpolation, %-formatting, str.format, concatenation, multi-hop aliasing, "
         "inter-procedural helpers, request-dictionary sources, loop-built WHERE clauses, ORM "
         "raw-SQL escape hatches, multi-statement execution, string.Template substitution, "
         "identifier injection in ORDER BY, fragment joining, partial parameterisation and "
         "ineffective sanitisers. Fifteen safe families cover the correct idioms, and eight of "
         "them are deliberate hard negatives: safe code that still interpolates a value into a "
         "SQL string via an integer cast, an allow-listed identifier, an enum fragment lookup, a "
         "regex-validated token, module constants, a composable-SQL API, SQL text that is never "
         "executed, or a non-SQL sink.")
    para(doc, "Controls keep the task honest. Vulnerable and safe samples draw identifiers, "
         "tables, columns, docstrings, decorators and comments from the same pools, so no name "
         "is predictive. Comments are adversarial, with “input is already sanitised” "
         "appearing on both classes. Five database APIs, seven wrapping styles, optional type "
         "hints and zero to three distractor statements per sample are all drawn independently "
         "of the label. Every sample is validated with ast.parse, giving a 100% syntactic "
         "validity rate, and the corpus is de-duplicated on the SHA-1 of its source.")
    table(doc, ["Property", "Value"], [
        ["Total functions / unique source", "60,000 / 60,000"],
        ["Families (vulnerable / safe)", "30 (15 / 15)"],
        ["Hard negatives", "14,834"],
        ["Class balance (safe / vulnerable)", "30,003 / 29,997"],
        ["Lines per function (mean / median / range)", "11.5 / 11 / 2" + EN + "27"],
        ["Annotated statements per function (mean)", "2.43"],
        ["Database APIs", "sqlite3, psycopg2, mysql.connector, pymysql, pyodbc"],
        ["Difficulty (easy / medium / hard)", "17,336 / 17,334 / 25,330"],
        ["Syntactic validity (ast.parse)", "100%"],
    ], [46, 54])
    caption(doc, "Table 3. Corpus summary.")
    para(doc, "Twenty-four families form the in-distribution pool, split 70/10/20 into train "
         "(36,402), validation (5,208) and test-ID (10,392). Six families " + EM + " three "
         "vulnerable (inter-procedural helper, string.Template, fragment joining) and three safe "
         "(enum fragment lookup, never-executed SQL, composable-SQL API) " + EM + " are withheld "
         "from training entirely and constitute test-OOD (7,998). All six are rated hard, and "
         "all three held-out safe families are hard negatives, so the held-out false-positive "
         "rate is a pure hard-negative measure and is not directly comparable with the "
         "in-distribution figure. Every split is balanced to a positive rate of 0.500.")
    para(doc, "A validity check was run on the risk that identifier anonymisation collapses "
         "distinct functions onto identical token sequences. Measured on the encoded tensors, "
         "99.93% of training sequences are unique, and only 0.14% of test-ID and 0.00% of "
         "test-OOD sequences coincide with a training sequence, so the test splits are not "
         "measuring memorisation.")

    sub(doc, "Tokenisation")
    para(doc, "Three decisions carry most of the inductive bias. Identifiers outside a fixed API "
         "lexicon are rewritten to <ID_k>, with k the order of first appearance within that "
         "sample; this destroys name memorisation while preserving aliasing, so a variable "
         "written on one line and read on another is the same symbol in both places. String "
         "literals are decomposed into structural markers, so that a quoted interpolation "
         "(<SQ> <FFIELD_BEG> <ID_k> <FFIELD_END> <SQ>), a question-mark placeholder (<QMARK>) "
         "and a %s placeholder (<PCT_FMT>) become distinct tokens; without this the vulnerable "
         "and safe idioms are nearly indistinguishable. A hand-written scanner is used rather "
         "than the standard tokenize module, because CPython 3.12 changed f-string tokenisation "
         "and the notebook must reproduce on another interpreter. The resulting vocabulary is "
         "415 types.")

    sub(doc, "Experimental setup and parameter settings")
    table(doc, ["Parameter", "Value", "Parameter", "Value"], [
        ["Sequence length L", "160 (head+tail trunc.)", "Optimiser", "AdamW [18]"],
        ["Encoder width d", "96", "Base learning rate", "2.5 × 10⁻³"],
        ["Dilations / recept. field", "1, 2, 4 / 15 tokens", "Schedule", "warm-up 10% + cosine"],
        ["Attention heads H", "4", "Weight decay", "1 × 10⁻⁴"],
        ["Search embedding", "96", "Gradient clipping", "1.0"],
        ["Dropout", "0.1", "Batch size", "128"],
        ["λ_a / λ_r / κ", "0.5 / 0.3 / 0.1", "Families per batch", "16"],
        ["Trainable parameters", "212,385", "Epochs / patience", "5 / 2"],
        ["Training subset", "12,000 (stratified)", "Seeds", "0, 1, 2"],
    ], [26, 25, 25, 24])
    caption(doc, "Table 4. Hyper-parameters. The receptive field was selected as described "
            "below.")
    para(doc, "The threshold τ and the early-stopping epoch are both chosen on the "
         "validation split; test-ID and test-OOD are read once, at the end. The receptive field "
         "was chosen by a dedicated ablation. Measuring the token distance between the first and "
         "last annotated statement gives a median span of 37 tokens in-distribution and 58 on "
         "held-out families, which predicts that a wide field is necessary. Testing "
         "R ∈ {15, 31, 63, 127} refuted that prediction: in-distribution F1 varied by only "
         "0.0013 while held-out F1 varied by 0.121, and a wider field correlated with a worse "
         "held-out false-positive rate (Spearman ρ = +0.80) rather than a better one. "
         "Because in-distribution validation F1 is saturated near 0.999 for every configuration, "
         "no legitimate selection signal distinguishes them, and choosing R by its held-out "
         "score would constitute selection on the test set. R was therefore selected on "
         "validation, where all four tie, with computational cost as the tie-breaker, giving "
         "dilations 1, 2, 4. The ablation is reported as inconclusive about which field is "
         "better.")

    figure(doc, "fig12_receptive_field.png", 6.1,
           "Figure 1. Left: cumulative distribution of taint-span width against the candidate "
           "receptive fields. Right: the measured effect of widening the field " + EM + " the "
           "prediction is not supported.")

    sub(doc, "Performance evaluation criteria")
    para(doc, "Writing TP, FP, TN and FN for the confusion-matrix entries at threshold τ "
         "and N for the number of evaluated samples:")
    eq(doc, "Accuracy = (TP+TN)/N,    Precision = TP/(TP+FP),    Recall = TP/(TP+FN),")
    eq(doc, "F₁ = 2·Precision·Recall/(Precision+Recall),    FPR = FP/(FP+TN),")
    eq(doc, "MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)).")
    para(doc, "Recall is what a security team values first, because a missed defect ships. The "
         "false-positive rate is the headline metric of this study, because the project's "
         "premise is that static analysers are too noisy to use rather than insufficiently "
         "sensitive: each spurious alert consumes review budget. MCC is reported because it "
         "remains informative under imbalance, taking +1 for perfect agreement, 0 for chance and "
         "−1 for total disagreement [19]. ROC-AUC and PR-AUC summarise ranking across "
         "thresholds and are undefined for the rule-based analysers, which emit a hard decision; "
         "those cells are marked " + EM + ".")
    para(doc, "For code search, a corpus item is relevant to a query when it belongs to the same "
         "vulnerability family, that is, realises the same taint pattern. With |Q| queries:")
    eq(doc, "MRR = (1/|Q|) Σ_q 1/rank_q,    "
            "Recall@k = (1/|Q|) Σ_q 1[rel. item in top k],")
    eq(doc, "MAP = (1/|Q|) Σ_q (1/R_q) Σ_{j: rel} P@j,    "
            "nDCG@k = Σ_j rel_j/log₂(j+1) ÷ IDCG@k.")
    para(doc, "MRR answers how far an analyst must scroll before the first useful hit; MAP and "
         "nDCG reward getting the whole ordering right, nDCG with an explicit position discount. "
         "Queries are evaluated leave-one-out against a stratified corpus of 6,000 items.")
    para(doc, "Attention quality is evaluated directly, which most vulnerability-detection work "
         "omits. With valid_i the non-padding positions, k_i = |m_i| and T_i the k_i "
         "most-attended tokens:")
    eq(doc, "AMR_i = Σ_t α_{i,t} m_{i,t},    "
            "Lift_i = AMR_i ÷ (|m_i| / |valid_i|),")
    eq(doc, "Pointing_i = 1[argmax_t α_{i,t} ∈ m_i],    "
            "IoU_i = |T_i ∩ m_i| / |T_i ∪ m_i|.")
    para(doc, "Lift divides out the mass a uniform map would place on the annotation, so a lift "
         "of 1 means no better than attending everywhere, and the figure is comparable across "
         "functions of differing length. Each neural variant is trained with three seeds; tables "
         "report mean ± standard deviation, and improvement claims are supported by a "
         "paired t-test and a Wilcoxon signed-rank test with Cohen's d for effect size.")

    sub(doc, "Software and hardware")
    para(doc, "All experiments ran on an x86-64 AMD CPU (reported by the platform as AMD64 "
         "Family 25 Model 80) with 12 logical cores and six "
         "PyTorch threads, under Windows 11 Pro (build 26200), using Python 3.12.0, PyTorch "
         "2.14.1 (CPU build) [20], NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1 "
         "and Matplotlib 3.11.2. No GPU was used. Corpus generation takes approximately 11 "
         "seconds; tokenising and encoding all 60,000 samples takes about 32 seconds and is "
         "cached. The full comparative study " + EM + " twelve neural training runs plus "
         "baselines, tables and figures " + EM + " completed in 56.2 minutes, with individual "
         "neural runs averaging 238 to 279 seconds. The static analysers require no training and "
         "run at 0.227 ms (M1) and 0.794 ms (M2) per function.")
    pagebreak(doc)


# --------------------------------------------------------------------------- #
def _results(doc) -> None:
    heading(doc, "Discussion of results", "50")
    para(doc, "Results are reported on two test sets. Test-ID contains unseen functions from "
         "the twenty-four families used in training; test-OOD contains six families withheld "
         "entirely. The two must be read as a pair: the first asks whether a method fits the "
         "problem, the second whether it learned anything transferable.")

    sub(doc, "Detection on in-distribution families")
    table(doc, ["Method", "Acc.", "F1", "FPR", "Recall", "Prec.", "ROC-AUC", "MCC"], [
        ["M1 static, pattern rules", "0.8234", "0.8474", "0.3333", "0.9802", "0.7462", EM,
         "0.6812"],
        ["M2 static, AST taint", "0.9167", "0.9231", "0.1667", "1.0000", "0.8571", EM, "0.8452"],
        ["M3 TF-IDF + LR", "1.0000", "1.0000", "0.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
        ["M4 CNN, mean pooling", "0.9946", "0.9946", "0.0042", "0.9934", "0.9958", "0.9999",
         "0.9892"],
        ["M5 CNN + unguided attn.", "0.9979", "0.9979", "0.0032", "0.9989", "0.9968", "0.9999",
         "0.9957"],
        ["M6 M5 + contrastive head", "0.9984", "0.9984", "0.0013", "0.9981", "0.9987", "0.9999",
         "0.9967"],
        ["M7 AGNCS (proposed)", "0.9992", "0.9992", "0.0009", "0.9992", "0.9991", "0.9998",
         "0.9983"],
    ], [27, 10, 10, 10, 10, 10, 12, 11], size=8)
    caption(doc, "Table 5. Detection on test-ID (10,392 samples). Neural rows are means over "
            "three seeds; standard deviations are below 0.007 throughout.")
    rich(doc, [("The in-distribution split cannot discriminate between methods, and should not "
                "be used to argue anything. ", True, False),
               ("TF-IDF with logistic regression attains a perfect 1.0000 on all six metrics "
                "across 10,392 samples, and every neural variant scores between 0.9946 and "
                "0.9992. The synthetic corpus is linearly separable in token n-gram space within "
                "the families from which it was built. That is a property of the corpus rather "
                "than a result about any model, and reporting it honestly is necessary: a reader "
                "shown only this table would conclude the problem is solved. Only the static "
                "analysers are genuinely challenged here, M1 at FPR 0.333 and M2 at 0.167.",
                False, False)])

    sub(doc, "Detection on held-out families")
    table(doc, ["Method", "Acc.", "F1", "FPR", "Recall", "Prec.", "ROC-AUC", "MCC"], [
        ["M1 static, pattern rules", "0.2403", "0.1624", "0.6667", "0.1473", "0.1810", EM,
         "−0.5286"],
        ["M2 static, AST taint", "0.6667", "0.7500", "0.6667", "1.0000", "0.6000", EM, "0.4472"],
        ["M3 TF-IDF + LR", "0.8025", "0.8286", "0.3488", "0.9537", "0.7326", "0.9300", "0.6349"],
        ["M4 CNN, mean pooling", "0.7604", "0.7142", "0.0785", "0.5992", "0.8860", "0.7880",
         "0.5510"],
        ["M5 CNN + unguided attn.", "0.5268", "0.6416", "0.7933", "0.8470", "0.5172", "0.5669",
         "0.0633"],
        ["M6 M5 + contrastive head", "0.4070", "0.5361", "0.9083", "0.7223", "0.4315", "0.3700",
         "−0.2349"],
        ["M7 AGNCS (proposed)", "0.6220", "0.6000", "0.3252", "0.5692", "0.6353", "0.5946",
         "0.2453"],
    ], [27, 10, 10, 10, 10, 10, 12, 11], size=8)
    caption(doc, "Table 6. Detection on test-OOD (7,998 samples, six withheld families). "
            "Standard deviations of F1 over three seeds: M3 ±0.011, M4 ±0.023, "
            "M5 ±0.023, M6 ±0.140, M7 ±0.049.")
    figure(doc, "fig1_detection_comparison.png", 6.3,
           "Figure 2. F1 (higher better) and false-positive rate (lower better) by method, "
           "in-distribution and held-out. Error bars are one standard deviation over three "
           "seeds.")
    rich(doc, [("Two findings dominate this table, and neither favours the proposed model. ",
                True, False),
               ("First, the classical baseline wins the detection task on unseen patterns: "
                "TF-IDF reaches F1 0.829 and ROC-AUC 0.930, against AGNCS's 0.600 and 0.595. "
                "Second, attention actively harms out-of-distribution detection here. The "
                "attention-free control M4 is the best neural detector by F1 (0.714), "
                "false-positive rate (0.079) and ROC-AUC (0.788). Adding unguided attention "
                "raises the held-out false-positive rate from 0.079 to 0.793, and adding the "
                "contrastive head on top pushes it to 0.908 with ROC-AUC 0.370 " + EM + " worse "
                "than chance. AGNCS does not overturn this; it is beaten by plain mean pooling "
                "(F1 difference −0.114, p = 0.101; FPR difference +0.247, p = 0.016 in "
                "M4's favour).", False, False)])
    para(doc, "The mechanism is consistent with the receptive-field ablation reported above. If "
         "the decisive evidence is a short local motif " + EM + " a quoted interpolation adjacent "
         "to SQL keywords " + EM + " then a lexical model captures it directly and has no "
         "long-range capacity with which to overfit, whereas an attentional model has the freedom "
         "to key on family-specific layout that does not transfer. The static analysers behave as "
         "the project's motivation predicts but differ sharply from one another: M2 misses "
         "nothing on either split, holding recall 1.000 including all three withheld vulnerable "
         "families, whereas M1 falls from recall 0.980 in-distribution to 0.147 on held-out "
         "families, because its rules enumerate surface forms instead of reasoning about flow. "
         "M1's negative MCC of −0.529 indicates systematic disagreement with the truth, not "
         "mere noise.")

    sub(doc, "Confusion matrices and the meaning of each error type")
    table(doc, ["Method / split", "TP", "FP", "TN", "FN", "Interpretation"], [
        ["M2 static taint, ID", "5,196", "866", "4,330", "0", "nothing missed, 866 false alarms"],
        ["M2 static taint, OOD", "3,999", "2,666", "1,333", "0", "nothing missed, 2,666 alarms"],
        ["M3 TF-IDF, ID", "5,196", "0", "5,196", "0", "perfect, and therefore uninformative"],
        ["M3 TF-IDF, OOD", "3,824", "1,546", "2,453", "175", "175 defects shipped"],
        ["M7 AGNCS, ID", "5,193", "5", "5,191", "3", "3 defects shipped, 5 false alarms"],
        ["M7 AGNCS, OOD", "2,460", "1,215", "2,784", "1,539", "1,539 defects shipped"],
    ], [24, 10, 10, 10, 10, 36])
    caption(doc, "Table 7. Confusion matrices at the validation-selected threshold (seed 0). "
            "Test-ID has 10,392 samples and test-OOD 7,998, each balanced 50/50.")
    figure(doc, "fig4_confusion.png", 6.2,
           "Figure 3. Confusion matrices, best static analyser versus the proposed model, on "
           "both splits. Cells show counts and row-normalised percentages.")
    rich(doc, [("The two error types are not symmetric in cost. ", True, False),
               ("A false negative is a SQL-injection defect the tool declares safe: it ships to "
                "production, and the scanner has actively provided false assurance. A false "
                "positive is safe code the tool flags: it costs an engineer the minutes needed "
                "to read the function and dismiss the alert. One breach is worth far more than "
                "one wasted review, which is why recall is examined first. But false positives "
                "aggregate, and a queue that is mostly noise is abandoned, after which the "
                "tool's effective recall becomes zero regardless of its measured recall. That "
                "is the asymmetry this project targets: M2 has FN = 0 on both splits, so it "
                "never provides false assurance, but its 2,666 held-out false alarms against "
                "1,333 correct clearances mean an analyst discards two alerts for every one "
                "worth reading. AGNCS reverses the trade in the wrong direction on held-out "
                "data, cutting false alarms to 1,215 but shipping 1,539 defects, which is the "
                "worse failure for a security tool and the clearest argument against deploying "
                "it standalone.", False, False)])

    sub(doc, "Error examples")
    para(doc, "The two characteristic AGNCS failures on held-out families, taken verbatim from "
         "the test set with the model's predicted probability (decision threshold 0.27):")
    rich(doc, [("False negative " + EM + " missed vulnerability (family V11, p = 0.003). ",
                True, False),
               ("A %-mapping substitution places caller-controlled text inside a quoted SQL "
                "slot. The model is confidently wrong, because the training families express "
                "interpolation as an f-string field or a + concatenation and never as a "
                "%-mapping, so the token pattern is unfamiliar:", False, False)])
    for ln in ["def search_employee_by_status(page_token):",
               "    cursor = get_cursor()",
               "    stmt = \"UPDATE messages SET owner_id = 'pending' "
               "WHERE username = '%(val)s'\" % {\"val\": page_token}",
               "    cursor.execute(stmt)",
               "    return cursor.fetchall()"]:
        code(doc, ln)
    para(doc, "", after=4)
    rich(doc, [("False positive " + EM + " safe code flagged (family S12, p = 0.998). ",
                True, False),
               ("A SQL-shaped string is built and returned for display; the function contains no "
                "cursor and no execute call anywhere. The model has learned the motif “a "
                "quoted interpolation near SQL keywords” and has no representation of the "
                "absence of a sink, so it fires with near-total confidence:", False, False)])
    for ln in ["def select_row(user_input: str):",
               "    explain_text = f\"UPDATE employees SET category_id = 'pending' "
               "WHERE sku = '{user_input}'\"",
               "    # documentation only: this string is rendered, never executed",
               "    return {\"table\": \"employees\", \"preview\": explain_text}"]:
        code(doc, ln)
    para(doc, "Both errors are failures of the local-motif strategy rather than of capacity: the "
         "first needs a token pattern absent from training, and the second needs a judgement "
         "about what the function does not contain. Neither is addressed by more data from the "
         "same generator.", before=6)

    sub(doc, "Where the false positives actually come from")
    table(doc, ["Method", "Hard neg. (ID)", "Easy neg. (ID)", "Hard neg. (OOD)"], [
        ["M1 static, pattern rules", "0.8000", "0.0000", "0.6667"],
        ["M2 static, AST taint", "0.4000", "0.0000", "0.6667"],
        ["M3 TF-IDF + LR", "0.0000", "0.0000", "0.3487"],
        ["M4 CNN, mean pooling", "0.0081", "0.0014", "0.0785"],
        ["M5 CNN + unguided attn.", "0.0062", "0.0011", "0.7933"],
        ["M6 M5 + contrastive head", "0.0032", "0.0000", "0.9083"],
        ["M7 AGNCS (proposed)", "0.0021", "0.0000", "0.3252"],
    ], [34, 22, 22, 22])
    caption(doc, "Table 8. False-positive rate split by negative difficulty. The held-out split "
            "contains no easy negatives by construction.")
    para(doc, "This table isolates the project's premise. Both static analysers are essentially "
         "perfect on ordinary safe code " + EM + " a parameterised query is never misjudged "
         + EM + " and fail only on negatives that look dangerous: M1 flags 80% of "
         "in-distribution hard negatives and M2 flags 40%. Their entire precision problem is "
         "concentrated there, which confirms that the review burden such tools impose comes from "
         "semantic constructs their rules cannot evaluate, not from crude errors. Every learned "
         "method clears in-distribution hard negatives almost completely, AGNCS best at 0.0021, "
         "which is the clearest in-distribution advantage of learning over rules in this study.")

    sub(doc, "Per-family diagnosis")
    table(doc, ["Held-out family", "Label", "M1", "M2", "M3", "M4", "M7"], [
        ["V13 fragment joining", "vuln.", "1.000", "0.000", "0.000", "0.971", "1.000"],
        ["S12 SQL never executed", "safe", "0.000", "0.000", "0.804", "0.130", "0.945"],
        ["V11 string.Template", "vuln.", "0.558", "0.000", "0.137", "0.174", "0.282"],
        ["S08 enum fragment lookup", "safe", "1.000", "1.000", "0.243", "0.062", "0.030"],
        ["V06 inter-proc. helper", "vuln.", "1.000", "0.000", "0.002", "0.057", "0.010"],
        ["S15 composable SQL API", "safe", "1.000", "1.000", "0.000", "0.044", "0.000"],
    ], [32, 10, 10, 10, 10, 10, 10])
    caption(doc, "Table 9. Error rate per held-out family, averaged over seeds. Values near "
            "1.000 denote systematic failure on that pattern.")
    rich(doc, [("The two methods' failures are complementary, which is the most actionable "
                "result in the report. ", True, False),
               ("M2 handles every withheld vulnerable family perfectly (V13, V11 and V06 all at "
                "0.000) because taint propagation does not care what syntactic form the flow "
                "takes, yet it fails completely on S08 and S15 (both 1.000), because judging "
                "whether an allow-list or a driver-level identifier-quoting API renders a value "
                "harmless requires semantic reasoning no propagation rule encodes. AGNCS is the "
                "mirror image: it clears S15 (0.000), S08 (0.030) and V06 (0.010), precisely the "
                "semantic cases, but misses V13 entirely (1.000) and over-flags S12 (0.945). A "
                "combination of the two would therefore be stronger than either, which is the "
                "basis of the recommendation made below.", False, False)])
    para(doc, "AGNCS's two failures are interpretable. V13 assembles the query from a list of "
         "fragments, so the quote characters that mark the injection sit in a different literal "
         "from the SQL keywords, placing the two cues outside a single 15-token receptive field. "
         "S12 builds a SQL-shaped string and never executes it; the model has learned the motif "
         "“quoted interpolation near SQL keywords” and has no reliable representation "
         "of the absence of a sink. Both are failures of the local-motif strategy that otherwise "
         "serves the model well.")

    sub(doc, "Code search")
    table(doc, ["Method", "MAP (ID)", "MRR (ID)", "MAP (OOD)", "MRR (OOD)", "R@1 (OOD)"], [
        ["M3 TF-IDF cosine", "0.3869", "0.8603", "0.7961", "0.9953", "0.9913"],
        ["M4 CNN, mean pooling", "0.1780", "0.5606", "0.4176", "0.8854", "0.8247"],
        ["M5 CNN + unguided attn.", "0.2881", "0.8232", "0.3827", "0.9405", "0.9124"],
        ["M6 M5 + contrastive head", "0.9962", "0.9975", "0.8612", "0.9881", "0.9831"],
        ["M7 AGNCS (proposed)", "0.9961", "0.9977", "0.9121", "0.9957", "0.9938"],
    ], [30, 14, 13, 14, 13, 12])
    caption(doc, "Table 10. Query-by-example code search; relevance is “same vulnerability "
            "family”. 1,500 queries against a stratified 6,000-item corpus, leave-one-out.")
    figure(doc, "fig8_retrieval.png", 6.3,
           "Figure 4. Retrieval quality by method, in-distribution and held-out.")
    rich(doc, [("This is where the proposed model succeeds unambiguously. ", True, False),
               ("AGNCS attains mean average precision 0.912 on held-out families against 0.796 "
                "for the strongest baseline, and 0.996 in-distribution against 0.387. The "
                "contrastive term is what creates a usable index at all: without it MAP is 0.178 "
                "(M4) and 0.288 (M5), and with it 0.996. Attention guidance then adds a further "
                "0.051 MAP on held-out families over M6. Against the attention-free control the "
                "improvement in held-out MRR is 0.110 with p = 0.0034 and d = 9.86; "
                "in-distribution it is 0.437 with p = 0.00068 and d = 22.2. Unlike detection, "
                "this is a capability neither static analyser possesses in any form.",
                False, False)])

    sub(doc, "Attention quality")
    table(doc, ["Method and split", "Mass on annot.", "Lift", "Pointing", "Top-k IoU"], [
        ["M5 unguided attention (ID)", "0.7377", "2.4941", "0.9119", "0.5056"],
        ["M6 + contrastive head (ID)", "0.6780", "2.2633", "0.9419", "0.4337"],
        ["M7 AGNCS (ID)", "0.9527", "3.2979", "0.9964", "0.9712"],
        ["M5 unguided attention (OOD)", "0.7397", "2.2279", "0.8772", "0.4583"],
        ["M6 + contrastive head (OOD)", "0.6199", "1.8828", "0.7665", "0.3815"],
        ["M7 AGNCS (OOD)", "0.7780", "2.3576", "0.7755", "0.6464"],
    ], [34, 19, 15, 16, 16])
    caption(doc, "Table 11. Agreement between the attention map and the statement-level "
            "annotation. A lift of 1.0 means no better than attending uniformly.")
    figure(doc, "fig6_attention_quality.png", 6.3,
           "Figure 5. Attention quality metrics. The dashed line marks uniform attention.")
    para(doc, "The guidance mechanism does what it was designed to do. In-distribution, the "
         "overlap between the k most-attended tokens and the annotated taint path rises from "
         "0.506 unguided to 0.971 guided, pointing-game accuracy from 0.912 to 0.996, and lift "
         "from 2.494 to 3.298 (p = 0.020, d = 4.00). The guidance term itself converges to a KL "
         "divergence of 0.084 nats from the annotation. The honest reference point, however, is "
         "M5 rather than uniform attention: unguided attention already achieves a lift near 2.5 "
         "without supervision, because the taint-carrying tokens are the predictive ones. The "
         "correct claim is therefore that guidance sharpens an already-informative map, not that "
         "it creates alignment from nothing.")
    figure(doc, "fig7_attention_V05_multi_hop_alias.png", 6.1,
           "Figure 6. Line-level attention on a multi-hop aliasing sample, unguided (left) "
           "versus guided (right). Orange bars mark the annotated taint path.")

    sub(doc, "Statistical significance")
    table(doc, ["Comparison", "Split", "Metric", "M7", "Baseline", "Diff", "p (t)", "d"], [
        ["M7 vs M6", "OOD", "FPR", "0.3252", "0.9083", "−0.5831", "0.011", "−5.42"],
        ["M7 vs M6", "ID", "att. lift", "3.2979", "2.2633", "+1.0345", "0.0015", "15.07"],
        ["M7 vs M6", "OOD", "att. lift", "2.3576", "1.8828", "+0.4748", "0.038", "2.90"],
        ["M7 vs M6", "OOD", "F1", "0.6000", "0.5361", "+0.0639", "0.346", "0.71"],
        ["M7 vs M5", "OOD", "FPR", "0.3252", "0.7933", "−0.4680", "0.013", "−4.99"],
        ["M7 vs M5", "OOD", "MRR", "0.9957", "0.9405", "+0.0552", "0.0017", "14.17"],
        ["M7 vs M4", "OOD", "F1", "0.6000", "0.7142", "−0.1142", "0.101", "−1.68"],
        ["M7 vs M4", "OOD", "FPR", "0.3252", "0.0785", "+0.2467", "0.016", "4.57"],
        ["M7 vs M4", "OOD", "MRR", "0.9957", "0.8854", "+0.1103", "0.0034", "9.86"],
    ], [17, 8, 14, 12, 12, 13, 11, 11])
    caption(doc, "Table 12. Paired comparisons across three matched seeds. With n = 3 the "
            "Wilcoxon signed-rank test cannot fall below p = 0.25, so the paired t-test and "
            "Cohen's d carry the evidence and p-values are indicative.")
    rich(doc, [("The significance tests delimit precisely what attention guidance contributes. ",
                True, False),
               ("Against M6 it reduces the held-out false-positive rate by 0.583 (p = 0.011) and "
                "raises in-distribution attention lift by 1.035 (p = 0.0015), both large "
                "effects. The held-out F1 improvement of 0.064 is not significant (p = 0.346) "
                "and must not be claimed. Against M4 the proposed model is significantly worse "
                "on held-out FPR (p = 0.016) and significantly better on retrieval (p = 0.0034). "
                "The defensible interpretation is specific: λ_a acts as a regulariser that "
                "makes the contrastive search objective compatible with detection. Bolting a "
                "retrieval head onto the encoder without guidance destroys held-out detection, "
                "driving ROC-AUC to 0.370; adding guidance retains near-perfect retrieval while "
                "returning detection to above chance.", False, False)])
    pagebreak(doc)


# --------------------------------------------------------------------------- #
def _critical(doc) -> None:
    heading(doc, "Critical evaluation", "5")
    para(doc, "Comparing the methods on their own terms, the two static analysers are separated "
         "by whether they model data flow. M1's surface rules give recall 0.980 in-distribution "
         "but 0.147 on unseen idioms, a collapse no threshold adjustment can repair; M2's "
         "propagation to a fixed point gives recall 1.000 on both splits. What M2 cannot do is "
         "decide whether a guard constrains a value, so its 0.400 and 0.667 hard-negative "
         "false-positive rates are irreducible within a purely syntactic framework. M3 against "
         "M4 is more surprising: a bag-of-n-grams model over the same tokens beats the learned "
         "encoder on held-out families (F1 0.829 versus 0.714), indicating that on this corpus "
         "feature engineering in the tokeniser, not architecture, supplies most of the signal.")
    para(doc, "Among the neural ablations, M4 against M5 is the most uncomfortable comparison: "
         "adding attention pooling improves in-distribution F1 marginally (0.9946 to 0.9979) "
         "while raising the held-out false-positive rate tenfold, from 0.079 to 0.793. Attention "
         "gives the model capacity to concentrate on cues that happen to be reliable in training "
         "and are not reliable beyond it. M5 against M6 shows that the contrastive head "
         "transforms retrieval (MAP 0.288 to 0.996) while further degrading held-out detection, "
         "and M6 against M7 shows that guidance repairs much of that damage. The proposed model "
         "is thus best understood not as a better detector but as the configuration that makes a "
         "joint detector-retriever viable.")
    para(doc, "Three limits on confidence deserve emphasis. In-distribution validation F1 "
         "saturates near 0.999, so early stopping selects an essentially arbitrary epoch and "
         "held-out performance is dominated by seed noise; the receptive-field study showed "
         "held-out F1 varying by 0.121 against 0.0013 in-distribution. Three seeds suffice for a "
         "mean and a paired t-test but not for the Wilcoxon test to reach conventional "
         "significance. And high attention mass on the taint path demonstrates that the pooling "
         "layer weights those tokens, not that the decision depended on them [13]. On cost, the "
         "static analysers require no training and run in under a millisecond per function, "
         "while each neural variant needs roughly four minutes of CPU training; for a team "
         "wanting recall today, M2 is the rational choice.")


def _conclusion(doc) -> None:
    heading(doc, "Conclusion and recommendation", "5")
    para(doc, "This project set out to test whether supervising a neural model's attention with "
         "statement-level taint annotations would reduce the false positives that make static "
         "SQL injection analysers burdensome. The premise was confirmed and the remedy only "
         "partly supported. A genuine AST taint analyser missed nothing on either test split, "
         "holding recall 1.000, yet flagged 40% of in-distribution and 67% of held-out hard "
         "negatives, confirming that the cost of such tools lies in precision on code that "
         "merely looks dangerous. Against that, attention-guided detection did not succeed: on "
         "withheld vulnerability families a TF-IDF classifier reached F1 0.829 against the "
         "proposed model's 0.600, and an attention-free control generalised better than any "
         "attentional variant. Attention guidance did deliver two significant effects " + EM + " "
         "held-out false positives fell from 0.908 to 0.325 and attention overlap with the true "
         "taint path rose from 0.506 to 0.971 " + EM + " and the code-search half of the system "
         "succeeded outright, reaching 0.912 mean average precision on unseen families against "
         "0.796 for the best baseline.")
    para(doc, "The usage recommendation follows from the complementary failure modes in Table "
         "9: neither method should replace the other. Taint analysis should run first, because it "
         "misses nothing; the guided embedding should then rank its alerts, because it clears "
         "precisely the semantic hard negatives on which taint analysis fails. The proposed model "
         "should not be deployed as a standalone detector on code unlike its training "
         "distribution.")
    para(doc, "Four improvements are recommended, in order of expected value. First, partition "
         "the families three ways rather than two, reserving four held-out families for model "
         "selection and four for reporting; without a held-out validation signal no "
         "architectural choice, including the receptive field, can be tuned honestly, and this "
         "single change would remove the main methodological constraint on the present results. "
         "Second, validate on mined real-world Python advisories, since the synthetic corpus "
         "cannot contain idioms its generator never encoded. Third, fine-tune GraphCodeBERT "
         "given GPU budget, and test whether guidance adds anything on top of pre-training [9]. "
         "Fourth, supply the model a program-dependence graph rather than a token sequence, "
         "making explicit the aliasing the tokeniser currently only hints at, which would "
         "directly address the V13 failure where two cues fall outside one receptive field. A "
         "gradient-based attribution study would also be needed before the attention maps are "
         "presented as explanations rather than as correlates.")


def _references(doc) -> None:
    heading(doc, "References", "5")
    refs = [
        "OWASP Foundation, “OWASP Top 10:2021 " + EN + " A03 Injection,” 2021. "
        "[Online]. Available: https://owasp.org/Top10/A03_2021-Injection/",
        "MITRE, “CWE-89: Improper Neutralization of Special Elements used in an SQL Command "
        "(‘SQL Injection’),” Common Weakness Enumeration. [Online]. Available: "
        "https://cwe.mitre.org/data/definitions/89.html",
        "PyCQA, “Bandit: a security linter for Python " + EN + " B608 "
        "hardcoded_sql_expressions,” 2024. [Online]. Available: "
        "https://bandit.readthedocs.io/",
        "Semgrep, “python.lang.security.audit.formatted-sql-query,” Semgrep Registry, "
        "2024. [Online]. Available: https://semgrep.dev/r/",
        "Z. Li, D. Zou, S. Xu, X. Ou, H. Jin, S. Wang, Z. Deng and Y. Zhong, “VulDeePecker: "
        "A Deep Learning-Based System for Vulnerability Detection,” in Proc. Network and "
        "Distributed System Security Symposium (NDSS), 2018.",
        "Y. Zhou, S. Liu, J. Siow, X. Du and Y. Liu, “Devign: Effective Vulnerability "
        "Identification by Learning Comprehensive Program Semantics via Graph Neural "
        "Networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2019.",
        "J. Fan, Y. Li, S. Wang and T. N. Nguyen, “A C/C++ Code Vulnerability Dataset with "
        "Code Changes and CVE Summaries,” in Proc. 17th Int. Conf. on Mining Software "
        "Repositories (MSR), 2020, pp. 508" + EN + "512.",
        "Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang "
        "and M. Zhou, “CodeBERT: A Pre-Trained Model for Programming and Natural "
        "Languages,” in Findings of EMNLP, 2020, pp. 1536" + EN + "1547.",
        "D. Guo, S. Ren, S. Lu, Z. Feng, D. Tang, S. Liu, L. Zhou, N. Duan, A. Svyatkovskiy, "
        "S. Fu, M. Tufano, S. K. Deng, C. Clement, D. Drain, N. Sundaresan, J. Yin, D. Jiang and "
        "M. Zhou, “GraphCodeBERT: Pre-training Code Representations with Data Flow,” in "
        "Proc. Int. Conf. on Learning Representations (ICLR), 2021.",
        "H. Husain, H.-H. Wu, T. Gazit, M. Allamanis and M. Brockschmidt, “CodeSearchNet "
        "Challenge: Evaluating the State of Semantic Code Search,” arXiv:1909.09436, 2019.",
        "P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu and "
        "D. Krishnan, “Supervised Contrastive Learning,” in Advances in Neural "
        "Information Processing Systems (NeurIPS), 2020.",
        "D. Bahdanau, K. Cho and Y. Bengio, “Neural Machine Translation by Jointly Learning "
        "to Align and Translate,” in Proc. Int. Conf. on Learning Representations (ICLR), "
        "2015.",
        "S. Jain and B. C. Wallace, “Attention is not Explanation,” in Proc. Conf. of "
        "the North American Chapter of the ACL (NAACL-HLT), 2019, pp. 3543" + EN + "3556.",
        "J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher and B. C. Wallace, "
        "“ERASER: A Benchmark to Evaluate Rationalized NLP Models,” in Proc. 58th "
        "Annual Meeting of the ACL, 2020, pp. 4443" + EN + "4458.",
        "National Institute of Standards and Technology, “Software Assurance Reference "
        "Dataset (SARD) and Juliet Test Suite,” NIST. [Online]. Available: "
        "https://samate.nist.gov/SARD/",
        "F. Yu and V. Koltun, “Multi-Scale Context Aggregation by Dilated "
        "Convolutions,” in Proc. Int. Conf. on Learning Representations (ICLR), 2016.",
        "Y. N. Dauphin, A. Fan, M. Auli and D. Grangier, “Language Modeling with Gated "
        "Convolutional Networks,” in Proc. 34th Int. Conf. on Machine Learning (ICML), "
        "2017, pp. 933" + EN + "941.",
        "I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. "
        "Int. Conf. on Learning Representations (ICLR), 2019.",
        "B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 "
        "phage lysozyme,” Biochimica et Biophysica Acta, vol. 405, no. 2, "
        "pp. 442" + EN + "451, 1975.",
        "A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning "
        "Library,” in Advances in Neural Information Processing Systems (NeurIPS), 2019.",
    ]
    for i, r in enumerate(refs, 1):
        para(doc, "[%d]  %s" % (i, r), size=9, after=4, indent_left=28, hanging=28)
