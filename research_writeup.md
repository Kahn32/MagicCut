# MagicCut: uncertainty-aware material grouping with active graph correction

## Abstract

Material Magic Wand retrieves material-consistent components of an untextured, pre-segmented mesh from one query part. We study whether its released 1152-D part representation can support structured binary inference and explicit positive/negative clarification without retraining the encoder. MagicCut calibrates per-part retrieval scores on a mesh-disjoint validation split, builds a k-nearest-neighbour graph over released part features, minimizes a Potts graph-cut energy, and converts uncertain decisions into binary user questions. On the locked 94-mesh held-out split (230 queries), frozen graph inference does not improve mean held-out F1 (0.4794 versus 0.7368). After three simulated binary corrections, mean F1 is 0.6222 and remains below Material Magic Wand; the paired difference is -0.1146 [-0.1812, -0.0551], with a mesh-bootstrap interval below zero. The corrections do repair part of the graph-induced loss: their paired improvement over zero-click MagicCut is 0.1428 [0.1195, 0.1664]. The locked experiment therefore rejects both the headline graph-superiority hypothesis and the predefined secondary success criterion. It shows only that feedback repairs part of a substantially degraded graph solution; it does not establish a new state of the art or a human-interaction advantage.

## 1. Problem statement

The input is an untextured mesh already decomposed into fine-grained parts and one selected query part. The output is the subset of parts intended to share a material with that query. This is intra-mesh grouping, not initial part segmentation, material recognition from texture, or cross-object retrieval.

Material-aware grouping is intrinsically ambiguous: geometry, context, and artistic intent can disagree. A single global distance threshold cannot express that a similar-looking piece should be excluded from the intended group. Our research question is therefore not only whether a graph improves a frozen retrieval decision, but whether calibrated uncertainty and explicit positive/negative constraints reduce correction effort.

## 2. Material Magic Wand

Material Magic Wand renders every part in three configurations: isolated part, part with context, and full object. A DINO-v3-small encoder processes each image; concatenation yields the 1152-D pre-projection feature used for retrieval. Training uses a 128-D projection and supervised contrastive loss to pull parts with the same within-mesh material ID together and push different-material parts apart. At inference, candidates are ranked by L1 distance to the query feature and grouped with a validation-selected tolerance.

The paper defines similarity as negative L1 distance but prints an inequality whose sign conflicts with its tolerance description. We do not silently choose between the two textual conventions: this implementation explicitly uses and tests `L1 distance <= threshold`, which matches the described behaviour that a larger distance tolerance selects more parts. The released deduplication mapping is applied so identical components share one representative embedding.

## 3. Proposed extension

MagicCut leaves the released encoder frozen. A validation-fitted logistic calibrator converts query-candidate evidence into probabilities. For each mesh, a symmetric 50-nearest-neighbour graph uses the isolated-part feature family; pairwise affinities use temperature factor 0.5. Binary grouping minimizes calibrated unary costs plus a Potts term with weight 0.25, while the query and later feedback become hard constraints.

After each inference step, an influence-aware acquisition rule combines calibration entropy and diversity from already queried parts (weights 0.5, 0.5; graph-degree weight 0.0). The selected candidate receives a simulated yes/no answer from benchmark ground truth and graph inference is rerun, for at most 3 clicks.

This is an exploratory composition of released retrieval features, graph inference, and active correction. We make no unsupported claim that this composition is the first possible method of its kind.

## 4. Research questions

1. Does validation-frozen graph inference improve grouping F1 over released-feature threshold retrieval?
2. Does calibration provide a useful and inexpensive uncertainty signal?
3. Does influence-aware binary clarification outperform random clarification on the validation/feasibility experiments?
4. How much does locked held-out grouping improve after one to three simulated corrections?
5. Which mesh-size, target-size, and class-imbalance regimes produce regressions or catastrophic failures?

## 5. Benchmark and protocol

We use the released 100-mesh, 241-query benchmark and all 25,348 released representative part renders. The project-defined split contains six validation meshes; the other 94 meshes are held out. This is not presented as the paper's 5-mesh/13-query split, whose exact identities are not included in the released metadata we audited. All thresholds, graph parameters, calibration, uncertainty choice, and acquisition weights are frozen using validation experiments before full evaluation.

Every archive is checked against its upstream LFS SHA-256 and expected topology. Every cached feature is checked for representative-ID coverage, finite FP32 values, exact shape `N x 1152`, and its recorded digest. The final evaluator refuses to run unless all 100 caches pass and refuses to overwrite its immutable raw result.

Macro precision, recall, and grouping F1 weight each query equally. Retrieval AP, PR AUC, R-precision, and Recall@20 use the shared frozen embedding ranking. Confidence intervals resample meshes, not individual queries, to preserve within-mesh dependence. We report directional evidence only when a 95% mesh-bootstrap interval excludes zero; we do not label this a formal significance test.

## 6. Locked quantitative results

Held-out 94-mesh results:

| Method | Precision | Recall | F1 | Inference ms |
|---|---:|---:|---:|---:|
| Material Magic Wand | 0.7275 | 0.8828 | 0.7368 | 0.0431 |
| Adaptive threshold | 0.9138 | 0.4706 | 0.5204 | 1073.7529 |
| Calibrated retrieval | 0.9138 | 0.4881 | 0.5060 | 0.0615 |
| MagicCut (0 clicks) | 0.9489 | 0.4398 | 0.4794 | 4.3429 |
| MagicCut (1 click) | 0.9571 | 0.4800 | 0.5265 | 3.1389 |
| MagicCut (2 clicks) | 0.9641 | 0.5329 | 0.5814 | 3.1965 |
| MagicCut (3 clicks) | 0.9674 | 0.5737 | 0.6222 | 3.3627 |

Material Magic Wand F1 with mesh-bootstrap 95% interval: 0.7368 [0.6988, 0.7755]. MagicCut F1: 0.4794 [0.4146, 0.5448]. Three-click active F1: 0.6222 [0.5600, 0.6811].

Frozen graph inference does not improve mean held-out F1 (0.4794 versus 0.7368). After three simulated binary corrections, mean F1 is 0.6222 and remains below Material Magic Wand; the paired difference is -0.1146 [-0.1812, -0.0551], with a mesh-bootstrap interval below zero. The corrections do repair part of the graph-induced loss: their paired improvement over zero-click MagicCut is 0.1428 [0.1195, 0.1664].

The common frozen embedding ranking has held-out mAP 0.8818, PR AUC 0.8685, R-precision 0.8348, and Recall@20 0.8102. Graph ranking AP in statistical analysis is explicitly decision-aware (selected partition first with calibrated probability only as a tie-break) and is not described as a native continuous graph score.

## 7. Ablations and gates

- The validation graph sweep covered k in {5, 10, 20, 50}, six pairwise weights, three affinity temperatures, and isolated-part/context/combined edges. The frozen choice is the configuration stated above.
- Gate 2 retained graph inference because it improved validation F1 and generated structured uncertainty, while the initial small held-out feasibility split showed a regression. That pre-registered caveat is preserved.
- Calibration entropy was retained over exact graph min-marginals because its error-detection/runtime trade-off was better in the feasibility study.
- Gate 3 selected influence-aware clarification because it beat random querying on both validation and the held-out feasibility subset. It did not uniformly beat uncertainty-only acquisition; that comparison is reported rather than hidden.
- Raw-geometry features were not retained: the audited Objaverse case showed no validation improvement across the tested geometry weights.
- The released small/medium/full images share one camera token, so they are context crops rather than independent views. Multiview variance was therefore rejected rather than fabricated.

## 8. Failure analysis

statistical analysis reports paired query deltas, mesh-bootstrap intervals, tertiles of mesh size, target size and positive-class fraction, and the ten largest graph regressions. Catastrophic counts are defined before inspection (`F1 = 0` and `F1 <= 0.20`). The success and failure panels are selected deterministically by the largest and smallest held-out MagicCut-minus-Material-Magic-Wand F1 deltas.

## 9. Limitations

- User answers are simulated from benchmark labels, not collected from artists; clicks-to-quality is therefore an oracle-labelled interaction simulation, not a human-subject result.
- Calibration and graph choices are assessed on one released benchmark. External generalization remains untested.
- The public project release includes a checkpoint and render benchmark but no complete inference implementation. Preprocessing and checkpoint loading were reconstructed and structurally audited, so differences from the authors' private pipeline may remain.
- The benchmark has one selected camera view per representative. The project cannot evaluate genuine multiview uncertainty from the released render archives.
- Exact FP32 encoding and kNN graph construction are costly for the largest meshes. The demo is intended for representative subsets that fit interactive latency budgets.
- Graph smoothing can propagate an incorrect unary decision across visually related but materially distinct parts; explicit negative feedback is particularly important in those cases.
- Ground-truth material grouping retains artistic ambiguity even after manual refinement.

## 10. Conclusion

Neither predefined success criterion is met on the locked held-out evaluation. The evidence should be stated exactly as follows: Frozen graph inference does not improve mean held-out F1 (0.4794 versus 0.7368). After three simulated binary corrections, mean F1 is 0.6222 and remains below Material Magic Wand; the paired difference is -0.1146 [-0.1812, -0.0551], with a mesh-bootstrap interval below zero. The corrections do repair part of the graph-induced loss: their paired improvement over zero-click MagicCut is 0.1428 [0.1195, 0.1664]. This is a useful negative result: validation gains did not generalize, degradation grows sharply with mesh size, and three oracle-labelled corrections recover only part of the loss. A stronger novelty or effectiveness claim would require an author-verified reproduction, redesigned scalable graph inference, external benchmarks, and a real user study.

## Primary sources

- Jain et al., Material Magic Wand, CVPR 2026: <https://arxiv.org/abs/2603.17370>
- Jain, Mirzaei, and Gilitschenski, GaussianCut, NeurIPS 2024: <https://arxiv.org/abs/2411.07555>
