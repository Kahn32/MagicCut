# Paper-grounding notes

These notes constrain the final write-up; they are not claims about MagicCut's
results.

## Material Magic Wand

Primary source: Umangi Jain, Vladimir Kim, Matheus Gadelha, Igor Gilitschenski,
and Zhiqin Chen, *Material Magic Wand: Material-Aware Grouping of 3D Parts in
Untextured Meshes*, CVPR 2026 / arXiv:2603.17370v1.

- Input: an untextured mesh already decomposed into fine-grained parts plus one
  query part. The task is grouping parts within that mesh that are likely to
  receive the same material; it is not initial part segmentation.
- Each part is represented by an isolated-part render, a part-with-context
  render, and a full-object render. The isolated and contextual renders share a
  selected low-occlusion viewpoint; the full-object camera follows the
  object-center-to-part-centroid direction.
- A DINO-v3-small backbone encodes the three renders. Concatenating their
  features gives the 1152-D pre-projection representation `x`; a two-layer MLP
  maps `x` to an L2-normalized 128-D contrastive representation `z`.
- Training uses supervised contrastive learning on within-mesh material IDs.
  The paper reports that the pre-projection feature `x`, rather than `z`, is
  better for retrieval.
- Released inference ranks candidates by negative L1 distance between 1152-D
  features and thresholds the distance for grouping. The paper's displayed
  set uses `s <= lambda` after defining `s = -distance`, while the surrounding
  prose says increasing the tolerance includes more parts. Those two statements
  have opposite sign conventions. This repository therefore states and tests
  its implementation directly as `L1 distance <= threshold`; it does not claim
  that the printed inequality resolves the inconsistency.
- Duplicate components are collapsed to representatives before encoding and
  expanded for the intended full-part grouping.
- The released benchmark has 100 meshes and 241 queries. The paper reports
  macro retrieval metrics and macro grouping F1, with each method's grouping
  threshold chosen on a small held-out validation split.
- The paper's multiple-click mechanism adds positive query exemplars. It does
  not model explicit negative corrections or structured uncertainty.
- Reported limitations include ambiguous artistic intent and failures when
  self-occlusion prevents clear contextual renders.

Source: <https://arxiv.org/html/2603.17370v1>

## GaussianCut

Primary source: Umangi Jain, Ashkan Mirzaei, and Igor Gilitschenski,
*GaussianCut: Interactive Segmentation via Graph Cut for 3D Gaussian
Splatting*, NeurIPS 2024 / arXiv:2411.07555.

GaussianCut motivates the structural idea, not a direct algorithmic
reproduction. It treats scene Gaussians as graph nodes and minimizes a
foreground/background graph-cut energy that combines user-derived unary
evidence with scene-based pairwise structure. MagicCut transfers the general
pattern - binary graph inference plus hard user constraints - to a different
representation and task: material-group candidates within one pre-segmented
mesh. It does not claim equivalence to GaussianCut's graph, energy, or 3DGS
setting.

Source: <https://arxiv.org/abs/2411.07555>
