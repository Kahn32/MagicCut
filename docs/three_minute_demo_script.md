# Three-minute MagicCut demonstration script

## 0:00-0:25 - Problem

"Material Magic Wand asks an artist to click one pre-segmented mesh part and retrieves other parts likely to share its material. The released encoder is strong, but one threshold cannot express all artistic intent."

## 0:25-0:55 - Frozen baseline

Select a small mesh and a query part. Point out red for the query, orange for selected, yellow for uncertain, and grey for rejected. Run grouping.

"The baseline and MagicCut use the same frozen 1152-D released features. MagicCut adds validation-fitted probabilities and a k-nearest-neighbour graph; no test-mesh tuning occurs."

## 0:55-1:35 - Clarification

Point to the blue-outlined recommended part. Answer yes or no, then repeat for up to three clicks.

"Each answer becomes a hard positive or negative graph constraint. The next question balances calibration entropy with diversity from parts already queried. The benchmark label supplies the answer in evaluation; this interface accepts a real user's answer."

## 1:35-2:15 - Result

Show the held-out table and click curve.

"On 230 queries from 94 held-out meshes, Material Magic Wand reaches F1 0.7368, no-click MagicCut reaches 0.4794, and three simulated corrections reach 0.6222. Frozen graph inference does not improve mean held-out F1 (0.4794 versus 0.7368)."

## 2:15-2:40 - Failure case

Open the deterministic failure panel.

"Graph smoothing can connect similar-looking parts that ground truth treats as different materials. This is why the project keeps explicit negative correction and does not claim graph inference always helps."

## 2:40-3:00 - Takeaway

"The locked result is negative: this graph construction does not generalize and three corrections do not recover the direct baseline. The useful outcome is a reproducible failure diagnosis and evidence that feedback repairs part of the graph error. The next step is an author-verified reproduction and a redesigned scalable graph before any artist study."
