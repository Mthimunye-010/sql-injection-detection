// Part 2: results, evaluation, conclusion, references -- and packs the .docx.
//   node scripts/report_build.js

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber,
  PageBreak, LevelFormat, TabStopType,
} = require("docx");
const H = require("./build_report.js");
const { P, RP, H1, H2, BULLET, NUM, CODE, EQ, CAPTION, table, figure,
        FONT, SZ, SZ_SMALL, CONTENT_W, ROOT } = H;
const { children, add } = require("./report_content.js");

// =========================================================== DISCUSSION
add(H1("Discussion of results", "50"));
add(P("Results are reported on two test sets. Test-ID contains unseen functions from the "
 + "twenty-four families used in training; test-OOD contains six families withheld entirely. The "
 + "two must be read as a pair: the first asks whether a method fits the problem, the second "
 + "whether it learned anything transferable."));

add(H2("Detection on in-distribution families"));
add(table(["Method", "F1", "FPR", "Recall", "Precision", "ROC-AUC", "MCC"],
  [["M1 static, pattern rules", "0.8474", "0.3333", "0.9802", "0.7462", "—", "0.6812"],
   ["M2 static, AST taint", "0.9231", "0.1667", "1.0000", "0.8571", "—", "0.8452"],
   ["M3 TF-IDF + LR", "1.0000", "0.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
   ["M4 CNN, mean pooling", "0.9946", "0.0042", "0.9934", "0.9958", "0.9999", "0.9892"],
   ["M5 CNN + unguided attn.", "0.9979", "0.0032", "0.9989", "0.9968", "0.9999", "0.9957"],
   ["M6 M5 + contrastive head", "0.9984", "0.0013", "0.9981", "0.9987", "0.9999", "0.9967"],
   ["M7 AGNCS (proposed)", "0.9992", "0.0009", "0.9992", "0.9991", "0.9998", "0.9983"]],
  [3000, 1100, 1100, 1100, 1200, 1100, 1000]));
add(CAPTION("Table 5. Detection on test-ID (10,392 samples). Neural rows are means over three "
 + "seeds; standard deviations are below 0.007 throughout and are omitted for space."));
add(RP([["The in-distribution split cannot discriminate between methods, and should not be used "
 + "to argue anything. ", { bold: true }], ["TF-IDF with logistic regression attains a perfect "
 + "1.0000 on all six metrics across 10,392 samples, and every neural variant scores between "
 + "0.9946 and 0.9992. The synthetic corpus is linearly separable in token n-gram space within "
 + "the families from which it was built. That is a property of the corpus rather than a result "
 + "about any model, and reporting it honestly is necessary: a reader shown only this table would "
 + "conclude the problem is solved. Only the static analysers are genuinely challenged here, M1 "
 + "at FPR 0.333 and M2 at 0.167."]]));

add(H2("Detection on held-out families"));
add(table(["Method", "F1", "FPR", "Recall", "Precision", "ROC-AUC", "MCC"],
  [["M1 static, pattern rules", "0.1624", "0.6667", "0.1473", "0.1810", "—", "−0.5286"],
   ["M2 static, AST taint", "0.7500", "0.6667", "1.0000", "0.6000", "—", "0.4472"],
   ["M3 TF-IDF + LR", "0.8286", "0.3488", "0.9537", "0.7326", "0.9300", "0.6349"],
   ["M4 CNN, mean pooling", "0.7142", "0.0785", "0.5992", "0.8860", "0.7880", "0.5510"],
   ["M5 CNN + unguided attn.", "0.6416", "0.7933", "0.8470", "0.5172", "0.5669", "0.0633"],
   ["M6 M5 + contrastive head", "0.5361", "0.9083", "0.7223", "0.4315", "0.3700", "−0.2349"],
   ["M7 AGNCS (proposed)", "0.6000", "0.3252", "0.5692", "0.6353", "0.5946", "0.2453"]],
  [3000, 1100, 1100, 1100, 1200, 1100, 1000]));
add(CAPTION("Table 6. Detection on test-OOD (7,998 samples from six withheld families). "
 + "Standard deviations over three seeds: M3 ±0.011 F1, M4 ±0.023, M5 ±0.023, "
 + "M6 ±0.140, M7 ±0.049."));
add(...figure("fig1_detection_comparison.png", 6.4,
  "Figure 1. F1 (higher better) and false-positive rate (lower better) by method, "
  + "in-distribution and held-out. Error bars are one standard deviation over three seeds."));
add(RP([["Two findings dominate this table, and neither favours the proposed model. ", { bold: true }],
  ["First, the classical baseline wins the detection task on unseen patterns: TF-IDF reaches F1 "
 + "0.829 and ROC-AUC 0.930, against AGNCS’s 0.600 and 0.595. Second, attention actively "
 + "harms out-of-distribution detection here. The attention-free control M4 is the best neural "
 + "detector by F1 (0.714), false-positive rate (0.079) and ROC-AUC (0.788). Adding unguided "
 + "attention raises the held-out false-positive rate from 0.079 to 0.793, and adding the "
 + "contrastive head on top pushes it to 0.908 with ROC-AUC 0.370 — worse than chance. "
 + "AGNCS does not overturn this; it is beaten by plain mean pooling (difference in F1 "
 + "−0.114, p = 0.101, and in FPR +0.247, p = 0.016 in M4’s favour)."]]));
add(P("The mechanism is consistent with the receptive-field ablation reported in the methods "
 + "section. If the decisive evidence is a short local motif — a quoted interpolation "
 + "adjacent to SQL keywords — then a lexical model captures it directly and has no "
 + "long-range capacity with which to overfit, whereas an attentional model has the freedom to "
 + "key on family-specific layout that does not transfer. The static analysers behave as the "
 + "project’s motivation predicts but differ sharply from one another: M2 misses nothing on "
 + "either split, holding recall 1.000 including all three withheld vulnerable families, whereas "
 + "M1 falls from recall 0.980 in-distribution to 0.147 on held-out families, because its rules "
 + "enumerate surface forms instead of reasoning about flow. M1’s negative MCC of "
 + "−0.529 indicates systematic disagreement with the truth, not mere noise."));

add(H2("Where the false positives actually come from"));
add(table(["Method", "Hard negatives (ID)", "Easy negatives (ID)", "Hard negatives (OOD)"],
  [["M1 static, pattern rules", "0.8000", "0.0000", "0.6667"],
   ["M2 static, AST taint", "0.4000", "0.0000", "0.6667"],
   ["M3 TF-IDF + LR", "0.0000", "0.0000", "0.3487"],
   ["M4 CNN, mean pooling", "0.0081", "0.0014", "0.0785"],
   ["M5 CNN + unguided attn.", "0.0062", "0.0011", "0.7933"],
   ["M6 M5 + contrastive head", "0.0032", "0.0000", "0.9083"],
   ["M7 AGNCS (proposed)", "0.0021", "0.0000", "0.3252"]],
  [3400, 2100, 2100, 2000]));
add(CAPTION("Table 7. False-positive rate split by negative difficulty. The held-out split "
 + "contains no easy negatives by construction, so that column is omitted for it."));
add(P("This table isolates the project’s premise. Both static analysers are essentially "
 + "perfect on ordinary safe code — a parameterised query is never misjudged — and "
 + "fail only on negatives that look dangerous: M1 flags 80% of in-distribution hard negatives "
 + "and M2 flags 40%. Their entire precision problem is concentrated there, which confirms that "
 + "the review burden such tools impose comes from semantic constructs their rules cannot "
 + "evaluate, not from crude errors. Every learned method clears in-distribution hard negatives "
 + "almost completely, AGNCS best at 0.0021, which is the clearest in-distribution advantage of "
 + "learning over rules in this study."));

add(H2("Per-family diagnosis"));
add(table(["Held-out family", "Label", "M1", "M2", "M3", "M4", "M7"],
  [["V13 fragment joining", "vuln.", "1.000", "0.000", "0.000", "0.971", "1.000"],
   ["S12 SQL never executed", "safe", "0.000", "0.000", "0.804", "0.130", "0.945"],
   ["V11 string.Template", "vuln.", "0.558", "0.000", "0.137", "0.174", "0.282"],
   ["S08 enum fragment lookup", "safe", "1.000", "1.000", "0.243", "0.062", "0.030"],
   ["V06 inter-proc. helper", "vuln.", "1.000", "0.000", "0.002", "0.057", "0.010"],
   ["S15 composable SQL API", "safe", "1.000", "1.000", "0.000", "0.044", "0.000"]],
  [3000, 900, 950, 950, 950, 950, 950]));
add(CAPTION("Table 8. Error rate per held-out family, averaged over seeds. "
 + "Values near 1.000 denote systematic failure on that pattern."));
add(...figure("fig5_family_errors_test_ood.png", 6.4,
  "Figure 2. Per-family error rate on held-out families for every method."));
add(RP([["The two methods’ failures are complementary, which is the most actionable result "
 + "in the report. ", { bold: true }], ["M2 handles every withheld vulnerable family perfectly "
 + "(V13, V11 and V06 all at 0.000) because taint propagation does not care what syntactic form "
 + "the flow takes, yet it fails completely on S08 and S15 (both 1.000), because judging whether "
 + "an allow-list or a driver-level identifier quoting API renders a value harmless requires "
 + "semantic reasoning no propagation rule encodes. AGNCS is the mirror image: it clears S15 "
 + "(0.000), S08 (0.030) and V06 (0.010), precisely the semantic cases, but misses V13 entirely "
 + "(1.000) and over-flags S12 (0.945). A combination of the two would therefore be stronger "
 + "than either, which is the basis of the recommendation made below."]]));
add(P("AGNCS’s two failures are interpretable. V13 assembles the query from a list of "
 + "fragments, so the quote characters that mark the injection sit in a different literal from "
 + "the SQL keywords, placing the two cues outside a single 15-token receptive field. S12 builds "
 + "a SQL-shaped string and never executes it; the model has learned the motif “quoted "
 + "interpolation near SQL keywords” and has no reliable representation of the absence of a "
 + "sink. Both are failures of the local-motif strategy that otherwise serves the model well."));

add(H2("Code search"));
add(table(["Method", "MAP (ID)", "MRR (ID)", "MAP (OOD)", "MRR (OOD)", "R@1 (OOD)"],
  [["M3 TF-IDF cosine", "0.3869", "0.8603", "0.7961", "0.9953", "0.9913"],
   ["M4 CNN, mean pooling", "0.1780", "0.5606", "0.4176", "0.8854", "0.8247"],
   ["M5 CNN + unguided attn.", "0.2881", "0.8232", "0.3827", "0.9405", "0.9124"],
   ["M6 M5 + contrastive head", "0.9962", "0.9975", "0.8612", "0.9881", "0.9831"],
   ["M7 AGNCS (proposed)", "0.9961", "0.9977", "0.9121", "0.9957", "0.9938"]],
  [3000, 1400, 1300, 1400, 1300, 1200]));
add(CAPTION("Table 9. Query-by-example code search; relevance is “same vulnerability "
 + "family”. 1,500 queries against a stratified 6,000-item corpus, leave-one-out."));
add(...figure("fig8_retrieval.png", 6.4,
  "Figure 3. Retrieval quality by method, in-distribution and held-out."));
add(RP([["This is where the proposed model succeeds unambiguously. ", { bold: true }], ["AGNCS "
 + "attains mean average precision 0.912 on held-out families against 0.796 for the strongest "
 + "baseline, and 0.996 in-distribution against 0.387. The contrastive term is what creates a "
 + "usable index at all: without it, MAP is 0.178 (M4) and 0.288 (M5), and with it 0.996. "
 + "Attention guidance then adds a further 0.051 MAP on held-out families over M6. Against the "
 + "attention-free control the improvement in held-out MRR is 0.110 with p = 0.0034 and "
 + "d₂ = 9.86; in-distribution it is 0.437 with p = 0.00068 and d₂ = 22.2. Unlike "
 + "detection, this is a capability neither static analyser possesses in any form."]]));

add(H2("Attention quality"));
add(table(["Method", "Mass on annotation", "Lift over uniform", "Pointing", "Top-k IoU"],
  [["M5 unguided attention (ID)", "0.7377", "2.4941", "0.9119", "0.5056"],
   ["M6 + contrastive head (ID)", "0.6780", "2.2633", "0.9419", "0.4337"],
   ["M7 AGNCS (ID)", "0.9527", "3.2979", "0.9964", "0.9712"],
   ["M5 unguided attention (OOD)", "0.7397", "2.2279", "0.8772", "0.4583"],
   ["M6 + contrastive head (OOD)", "0.6199", "1.8828", "0.7665", "0.3815"],
   ["M7 AGNCS (OOD)", "0.7780", "2.3576", "0.7755", "0.6464"]],
  [3400, 1900, 1700, 1300, 1300]));
add(CAPTION("Table 10. Agreement between the attention map and the statement-level annotation. "
 + "A lift of 1.0 means no better than attending uniformly."));
add(...figure("fig6_attention_quality.png", 6.4,
  "Figure 4. Attention quality metrics. The dashed line marks uniform attention."));
add(P("The guidance mechanism does what it was designed to do. In-distribution, the overlap "
 + "between the k most-attended tokens and the annotated taint path rises from 0.506 unguided to "
 + "0.971 guided, pointing-game accuracy from 0.912 to 0.996, and lift from 2.494 to 3.298 "
 + "(p = 0.020, d₂ = 4.00). The guidance term itself converges to a KL divergence of 0.084 "
 + "nats from the annotation. The honest reference point, however, is M5 rather than uniform "
 + "attention: unguided attention already achieves a lift near 2.5 without supervision, because "
 + "the taint-carrying tokens are the predictive ones. The correct claim is therefore that "
 + "guidance sharpens an already-informative map, not that it creates alignment from nothing."));
add(...figure("fig7_attention_V05_multi_hop_alias.png", 6.2,
  "Figure 5. Line-level attention on a multi-hop aliasing sample, unguided (left) versus guided "
  + "(right). Orange bars mark the annotated taint path."));

add(H2("Statistical significance"));
add(table(["Comparison", "Split", "Metric", "Proposed", "Baseline", "Diff", "p (t)", "d₂"],
  [["M7 vs M6", "OOD", "FPR", "0.3252", "0.9083", "−0.5831", "0.011", "−5.42"],
   ["M7 vs M6", "ID", "att. lift", "3.2979", "2.2633", "+1.0345", "0.0015", "15.07"],
   ["M7 vs M6", "OOD", "att. lift", "2.3576", "1.8828", "+0.4748", "0.038", "2.90"],
   ["M7 vs M6", "OOD", "F1", "0.6000", "0.5361", "+0.0639", "0.346", "0.71"],
   ["M7 vs M5", "OOD", "FPR", "0.3252", "0.7933", "−0.4680", "0.013", "−4.99"],
   ["M7 vs M5", "OOD", "MRR", "0.9957", "0.9405", "+0.0552", "0.0017", "14.17"],
   ["M7 vs M4", "OOD", "F1", "0.6000", "0.7142", "−0.1142", "0.101", "−1.68"],
   ["M7 vs M4", "OOD", "FPR", "0.3252", "0.0785", "+0.2467", "0.016", "4.57"],
   ["M7 vs M4", "OOD", "MRR", "0.9957", "0.8854", "+0.1103", "0.0034", "9.86"]],
  [1700, 800, 1300, 1200, 1200, 1200, 1100, 1100]));
add(CAPTION("Table 11. Paired comparisons across three matched seeds. With n = 3 the Wilcoxon "
 + "signed-rank test cannot fall below p = 0.25, so the paired t-test and d₂ carry the "
 + "evidence and p-values should be read as indicative."));
add(RP([["The significance tests delimit precisely what attention guidance contributes. ",
  { bold: true }], ["Against M6 it reduces the held-out false-positive rate by 0.583 "
 + "(p = 0.011) and raises in-distribution attention lift by 1.035 (p = 0.0015), both large "
 + "effects. The held-out F1 improvement of 0.064 is not significant (p = 0.346) and must not be "
 + "claimed. Against M4 the proposed model is significantly worse on held-out FPR (p = 0.016) and "
 + "significantly better on retrieval (p = 0.0034). The defensible interpretation is specific: "
 + "λ_a acts as a regulariser that makes the contrastive search objective compatible with "
 + "detection. Bolting a retrieval head onto the encoder without guidance destroys held-out "
 + "detection, driving ROC-AUC to 0.370; adding guidance retains near-perfect retrieval while "
 + "returning detection to above chance."]]));
add(...figure("fig12_receptive_field.png", 6.4,
  "Figure 6. Left: cumulative distribution of taint-span width against candidate receptive "
  + "fields. Right: measured effect of widening the field — the prediction fails."));
add(new Paragraph({ children: [new PageBreak()] }));

// =========================================================== CRITICAL EVALUATION
add(H1("Critical evaluation", "5"));
add(P("Comparing the methods on their own terms, the two static analysers are separated by "
 + "whether they model data flow. M1’s surface rules give recall 0.980 in-distribution but "
 + "0.147 on unseen idioms, a collapse that no threshold adjustment can repair; M2’s "
 + "propagation to a fixed point gives recall 1.000 on both splits. What M2 cannot do is decide "
 + "whether a guard constrains a value, so its 0.400 and 0.667 hard-negative false-positive "
 + "rates are irreducible within a purely syntactic framework. M3 against M4 is more surprising: "
 + "a bag-of-n-grams model over the same tokens beats the learned encoder on held-out families "
 + "(F1 0.829 versus 0.714), indicating that on this corpus feature engineering in the tokeniser, "
 + "not architecture, supplies most of the signal."));
add(P("Among the neural ablations, M4 against M5 is the most uncomfortable comparison: adding "
 + "attention pooling improves in-distribution F1 marginally (0.9946 to 0.9979) while raising "
 + "the held-out false-positive rate tenfold, from 0.079 to 0.793. Attention gives the model "
 + "capacity to concentrate on cues that happen to be reliable in training and are not reliable "
 + "beyond it. M5 against M6 shows that the contrastive head transforms retrieval (MAP 0.288 to "
 + "0.996) while further degrading held-out detection, and M6 against M7 shows that guidance "
 + "repairs much of that damage. The proposed model is thus best understood not as a better "
 + "detector but as the configuration that makes a joint detector-retriever viable."));
add(P("Three limits on confidence deserve emphasis. In-distribution validation F1 saturates near "
 + "0.999, so early stopping selects an essentially arbitrary epoch and held-out performance is "
 + "dominated by seed noise; the receptive-field study showed held-out F1 varying by 0.121 "
 + "against 0.0013 in-distribution. Three seeds suffice for a mean and a paired t-test but not "
 + "for the Wilcoxon test to reach conventional significance. And high attention mass on the "
 + "taint path demonstrates that the pooling layer weights those tokens, not that the decision "
 + "depended on them [13]. On cost, the static analysers require no training and run in under a "
 + "millisecond per function, while each neural variant needs roughly four minutes of CPU "
 + "training; for a team wanting recall today, M2 is the rational choice."));

// =========================================================== CONCLUSION
add(H1("Conclusion and recommendation", "5"));
add(P("This project set out to test whether supervising a neural model’s attention with "
 + "statement-level taint annotations would reduce the false positives that make static SQL "
 + "injection analysers burdensome. The premise was confirmed and the remedy only partly "
 + "supported. A genuine AST taint analyser missed nothing on either test split, holding recall "
 + "1.000, yet flagged 40% of in-distribution and 67% of held-out hard negatives, confirming that "
 + "the cost of such tools lies in precision on code that merely looks dangerous. Against that, "
 + "attention-guided detection did not succeed: on withheld vulnerability families a TF-IDF "
 + "classifier reached F1 0.829 against the proposed model’s 0.600, and an attention-free "
 + "control generalised better than any attentional variant. Attention guidance did deliver two "
 + "significant effects — held-out false positives fell from 0.908 to 0.325 and attention "
 + "overlap with the true taint path rose from 0.506 to 0.971 — and the code-search half of "
 + "the system succeeded outright, reaching 0.912 mean average precision on unseen families "
 + "against 0.796 for the best baseline."));
add(P("The recommendation on usage follows from the complementary failure modes in Table 8. "
 + "Neither method should replace the other. Taint analysis should run first, because it misses "
 + "nothing; the guided embedding should then rank and cluster its alerts, because it clears "
 + "precisely the semantic hard negatives — allow-lists, composable-SQL APIs, enum lookups "
 + "— on which taint analysis fails. The deliverable is an ordering of a high-recall alert "
 + "queue, which is what an analyst’s time actually requires. The proposed model should not "
 + "be deployed as a standalone detector on code unlike its training distribution."));
add(P("Four improvements are recommended, in order of expected value. First, partition the "
 + "families three ways rather than two, reserving four held-out families for model selection and "
 + "four for reporting; without a held-out validation signal no architectural choice, including "
 + "the receptive field, can be tuned honestly, and this single change would remove the main "
 + "methodological constraint on the present results. Second, validate on mined real-world "
 + "Python advisories, since the synthetic corpus cannot contain idioms its generator never "
 + "encoded. Third, fine-tune GraphCodeBERT given GPU budget, and test whether guidance adds "
 + "anything on top of pre-training [9]. Fourth, supply the model a program-dependence graph "
 + "rather than a token sequence, making explicit the aliasing the tokeniser currently only "
 + "hints at, which would directly address the V13 failure where two cues fall outside one "
 + "receptive field. A gradient-based attribution study would also be needed before the attention "
 + "maps are presented as explanations rather than as correlates."));

// =========================================================== REFERENCES
add(H1("References", "5"));
const refs = [
  "OWASP Foundation, “OWASP Top 10:2021 – A03 Injection,” 2021. [Online]. Available: https://owasp.org/Top10/A03_2021-Injection/",
  "MITRE, “CWE-89: Improper Neutralization of Special Elements used in an SQL Command (‘SQL Injection’),” Common Weakness Enumeration. [Online]. Available: https://cwe.mitre.org/data/definitions/89.html",
  "PyCQA, “Bandit: a security linter for Python – B608 hardcoded_sql_expressions,” 2024. [Online]. Available: https://bandit.readthedocs.io/",
  "Semgrep, “python.lang.security.audit.formatted-sql-query,” Semgrep Registry, 2024. [Online]. Available: https://semgrep.dev/r/python.lang.security.audit.formatted-sql-query",
  "Z. Li, D. Zou, S. Xu, X. Ou, H. Jin, S. Wang, Z. Deng and Y. Zhong, “VulDeePecker: A Deep Learning-Based System for Vulnerability Detection,” in Proc. Network and Distributed System Security Symposium (NDSS), 2018.",
  "Y. Zhou, S. Liu, J. Siow, X. Du and Y. Liu, “Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2019.",
  "J. Fan, Y. Li, S. Wang and T. N. Nguyen, “A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries,” in Proc. 17th Int. Conf. on Mining Software Repositories (MSR), 2020, pp. 508–512.",
  "Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang and M. Zhou, “CodeBERT: A Pre-Trained Model for Programming and Natural Languages,” in Findings of EMNLP, 2020, pp. 1536–1547.",
  "D. Guo, S. Ren, S. Lu, Z. Feng, D. Tang, S. Liu, L. Zhou, N. Duan, A. Svyatkovskiy, S. Fu, M. Tufano, S. K. Deng, C. Clement, D. Drain, N. Sundaresan, J. Yin, D. Jiang and M. Zhou, “GraphCodeBERT: Pre-training Code Representations with Data Flow,” in Proc. Int. Conf. on Learning Representations (ICLR), 2021.",
  "H. Husain, H.-H. Wu, T. Gazit, M. Allamanis and M. Brockschmidt, “CodeSearchNet Challenge: Evaluating the State of Semantic Code Search,” arXiv:1909.09436, 2019.",
  "P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu and D. Krishnan, “Supervised Contrastive Learning,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.",
  "D. Bahdanau, K. Cho and Y. Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate,” in Proc. Int. Conf. on Learning Representations (ICLR), 2015.",
  "S. Jain and B. C. Wallace, “Attention is not Explanation,” in Proc. Conf. of the North American Chapter of the ACL (NAACL-HLT), 2019, pp. 3543–3556.",
  "J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher and B. C. Wallace, “ERASER: A Benchmark to Evaluate Rationalized NLP Models,” in Proc. 58th Annual Meeting of the ACL, 2020, pp. 4443–4458.",
  "National Institute of Standards and Technology, “Software Assurance Reference Dataset (SARD) and Juliet Test Suite,” NIST. [Online]. Available: https://samate.nist.gov/SARD/",
  "F. Yu and V. Koltun, “Multi-Scale Context Aggregation by Dilated Convolutions,” in Proc. Int. Conf. on Learning Representations (ICLR), 2016.",
  "Y. N. Dauphin, A. Fan, M. Auli and D. Grangier, “Language Modeling with Gated Convolutional Networks,” in Proc. 34th Int. Conf. on Machine Learning (ICML), 2017, pp. 933–941.",
  "I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. Int. Conf. on Learning Representations (ICLR), 2019.",
  "B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” Biochimica et Biophysica Acta, vol. 405, no. 2, pp. 442–451, 1975.",
  "A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in Advances in Neural Information Processing Systems (NeurIPS), 2019.",
];
refs.forEach((r, i) => add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100, line: 240 },
  indent: { left: 567, hanging: 567 },
  children: [new TextRun({ text: `[${i + 1}]\t${r}`, font: FONT, size: SZ_SMALL })],
  tabStops: [{ type: TabStopType.LEFT, position: 567 }],
})));

// =========================================================== PACK
const doc = new Document({
  creator: "NCOS734 PG4 Group",
  title: "Attention-Guided Neural Code Search for SQL Injection Detection",
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 240 } } } }] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 240 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1008, right: 1152, bottom: 900, left: 1152,
                  header: 706, footer: 706 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
        children: [
          new TextRun({ text: "Computer Science & IT Department", font: FONT, size: 16 }),
          new TextRun({ text: "\tLecturer: MA Arasomwan  |  Page ", font: FONT, size: 16 }),
          new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 }),
        ],
      })] }),
    },
    children,
  }],
});

const out = path.join(ROOT, "NCOS734_PG4_Report.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log(`wrote ${path.basename(out)}  (${(buf.length / 1024).toFixed(0)} KB, `
            + `${children.length} block elements)`);
});
