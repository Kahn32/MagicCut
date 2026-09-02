# Prompts 1-3 Audit

This report records the live outputs of the official-resource audit on
2026-08-27. Generated machine-readable evidence lives in `reports/generated/`.
Large downloaded assets remain under ignored `data/` directories.

## Model checkpoint audit

- Status: **passed for provenance and structure**.
- Official model revision: `cfac3017592ab6925f3efe46ab90e53027c3ba45`.
- `checkpoint.pt`: 90,249,468 bytes; SHA-256
  `d86f28e222584d25be3d0d5017de60a88bf45b574d2bf5c73e12010ecfeefe97`.
  Both match the pinned official resource.
- The safe load yields a plain `OrderedDict` with 179 tensors: 175 under
  `backbone` and 4 under `head`.
- Structural evidence: a 12-block, 384-dimensional ViT with 14x14 patch
  projection, positional shape `[1, 1370, 384]`, and a projection head
  `1152 -> 384 -> 128`.
- Gate remaining: the authors' public GitHub repository contains project
  materials but no inference implementation. Exact preprocessing and feature
  concatenation must therefore be reconstructed and tested before any baseline
  metric is claimed.

## Benchmark metadata audit

- Status: **passed with documented release anomalies**.
- Repository: 446 files, including 241 labels, 100 deduplicated render archives,
  100 full render archives, two metadata files, and three top-level/support
  files.
- `objaverse_uids.json`: 100 unique UIDs. Labels: 241 queries across 100 meshes,
  1-8 queries per mesh. Every primary query occurs in its final selection.
- Target selection size ranges from 2 to 32,267; median 20.
- 174 queries include `original_selection`; two official labels omit
  `material_id`. The parser preserves these as `None` and reports the anomaly.
- The UID list and label folders differ by three substitutions in each
  direction. Exact IDs are retained in `benchmark_audit.json`; none are silently
  discarded.
- Both render metadata files parse into 100 meshes and 25,348 representative
  mappings; their representative-to-duplicate mappings are equal.

## One-mesh pipeline check

- Status: **passed**.
- Selected smallest official deduplicated archive:
  `01c81b9173d9464aa8f3cb343432cbb7.tar.gz` (189,500 bytes).
- Secure extraction yielded 36 PNGs: 12 representative parts, each at full,
  medium, and small sizes, all in released view `0`.
- Both mesh queries (`3` and `18`) resolve to available rendered
  representatives. Query 3's final IDs `[3,4,7,9]` reduce to representatives
  `[3,7]`; query 18's `[18,19,20,21]` reduce to `[18,20]`.
- Visual inspection of `one_mesh_contact_sheet.png` confirms the query and
  target representatives are distinct rendered components of the same object.

## Deliberately unresolved

- Exact DINO preprocessing and the checkpoint's 1152-dimensional feature
  construction.
- The paper's threshold inequality/sign inconsistency.
- The exact 5-mesh/13-query validation split, unless disclosed in released
  metadata.
- Baseline reproduction metrics.
- Whether graph inference improves retrieval or grouping.
