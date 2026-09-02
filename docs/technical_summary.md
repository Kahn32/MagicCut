# MagicCut - one-page technical summary

## Question

Can released Material Magic Wand features support structured grouping and efficient positive/negative correction without retraining the encoder?

## Method

Frozen 1152-D three-render features -> validation-fitted calibration -> kNN Potts graph cut -> influence-aware binary clarification. Query and feedback labels are hard graph constraints; up to three answers are simulated from benchmark ground truth.

## Protocol

- Released benchmark: 100 meshes, 241 queries, 25,348 representative parts.
- Project split: 6 validation meshes / 94 held-out meshes; all tuning uses validation only.
- Exact FP32 cache with archive, checkpoint, representative-ID, shape, finite-value, and SHA-256 checks.
- Macro query metrics; 95% bootstrap intervals resample meshes.

## Principal held-out results

| Method | Precision | Recall | F1 | Inference ms |
|---|---:|---:|---:|---:|
| Material Magic Wand | 0.7275 | 0.8828 | 0.7368 | 0.0431 |
| Adaptive threshold | 0.9138 | 0.4706 | 0.5204 | 1073.7529 |
| Calibrated retrieval | 0.9138 | 0.4881 | 0.5060 | 0.0615 |
| MagicCut (0 clicks) | 0.9489 | 0.4398 | 0.4794 | 4.3429 |
| MagicCut (1 click) | 0.9571 | 0.4800 | 0.5265 | 3.1389 |
| MagicCut (2 clicks) | 0.9641 | 0.5329 | 0.5814 | 3.1965 |
| MagicCut (3 clicks) | 0.9674 | 0.5737 | 0.6222 | 3.3627 |

Frozen graph inference does not improve mean held-out F1 (0.4794 versus 0.7368). After three simulated binary corrections, mean F1 is 0.6222 and remains below Material Magic Wand; the paired difference is -0.1146 [-0.1812, -0.0551], with a mesh-bootstrap interval below zero. The corrections do repair part of the graph-induced loss: their paired improvement over zero-click MagicCut is 0.1428 [0.1195, 0.1664].

## Honest interpretation

Neither success criterion is met: graph inference is substantially worse than direct retrieval, and three simulated corrections do not recover the baseline. The defensible contribution is a reproducible diagnosis of validation-to-test failure and a measured partial repair from explicit feedback. Answers are simulated, this is one benchmark, and the released authors' preprocessing code is unavailable.
