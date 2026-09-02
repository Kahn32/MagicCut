#!/usr/bin/env python3
"""Generate the final research narrative and outreach package from audited results."""

from __future__ import annotations

import json
import re
from pathlib import Path

from magiccut.frozen import GRAPH, INTERACTION
from magiccut.io import read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def f(value: float) -> str:
    return f"{value:.4f}"


def interval(value: dict) -> str:
    return f"{f(value['estimate'])} [{f(value['ci_95_low'])}, {f(value['ci_95_high'])}]"


def result_table(summary: dict) -> str:
    rows = []
    names = {
        "material_magic_wand": "Material Magic Wand",
        "adaptive": "Adaptive threshold",
        "calibrated": "Calibrated retrieval",
        "magiccut": "MagicCut (0 clicks)",
    }
    for key, label in names.items():
        row = summary["methods"][key]
        rows.append(f"| {label} | {f(row['precision'])} | {f(row['recall'])} | {f(row['f1'])} | {f(row['latency_ms'])} |")
    for row in summary["active"][1:]:
        rows.append(f"| MagicCut ({row['click']} click{'s' if row['click'] != 1 else ''}) | {f(row['precision'])} | {f(row['recall'])} | {f(row['f1'])} | {f(row['inference_ms'])} |")
    return "\n".join(
        ["| Method | Precision | Recall | F1 | Inference ms |", "|---|---:|---:|---:|---:|", *rows]
    )


def conclusions(summary: dict, delta_ci: dict, correction_ci: dict) -> tuple[str, str]:
    mmw = summary["methods"]["material_magic_wand"]["f1"]
    cut = summary["methods"]["magiccut"]["f1"]
    active3 = summary["active"][3]["f1"]
    if cut > mmw:
        graph = f"Frozen graph inference improves mean held-out F1 from {f(mmw)} to {f(cut)}."
    else:
        graph = f"Frozen graph inference does not improve mean held-out F1 ({f(cut)} versus {f(mmw)})."
    if delta_ci["ci_95_low"] > 0:
        active = (
            f"After three simulated binary corrections, mean F1 is {f(active3)}; the paired improvement over "
            f"Material Magic Wand is {interval(delta_ci)}, with a mesh-bootstrap interval above zero."
        )
    elif delta_ci["ci_95_high"] < 0:
        active = (
            f"After three simulated binary corrections, mean F1 is {f(active3)} and remains below Material Magic Wand; "
            f"the paired difference is {interval(delta_ci)}, with a mesh-bootstrap interval below zero. "
            f"The corrections do repair part of the graph-induced loss: their paired improvement over zero-click "
            f"MagicCut is {interval(correction_ci)}."
        )
    else:
        active = (
            f"After three simulated binary corrections, mean F1 is {f(active3)}; the paired improvement over "
            f"Material Magic Wand is {interval(delta_ci)}. The interval includes zero, so no directional evidence claim is made."
        )
    return graph, active


def main() -> None:
    report35 = read_json(ROOT / "reports/generated/locked_evaluation/report.json")
    report36 = read_json(ROOT / "reports/generated/statistics/report.json")
    figure_manifest = read_json(ROOT / "reports/generated/figures/manifest.json")
    raw_path = ROOT / "reports/generated/locked_evaluation/raw_results.json"
    if sha256_file(raw_path) != report35["raw_results_sha256"]:
        raise ValueError("locked evaluation raw digest mismatch")
    if report36["source_raw_results_sha256"] != report35["raw_results_sha256"]:
        raise ValueError("statistical analysis provenance mismatch")
    if figure_manifest["source_raw_results_sha256"] != report35["raw_results_sha256"]:
        raise ValueError("figure generation provenance mismatch")

    heldout = report35["heldout_94_meshes"]
    full = report35["full_100_meshes"]
    ci35 = report35["bootstrap_95_ci_by_mesh"]["heldout_94_meshes"]
    ci36 = report36["heldout_94_meshes"]["mesh_bootstrap_95_ci"]
    active_delta = ci36["active_3_minus_material_magic_wand.f1"]
    correction_delta = ci36["active_3_minus_magiccut.f1"]
    graph_text, active_text = conclusions(heldout, active_delta, correction_delta)
    graph_text_inline = graph_text[0].lower() + graph_text[1:]
    table = result_table(heldout)
    mmw = heldout["methods"]["material_magic_wand"]["f1"]
    cut = heldout["methods"]["magiccut"]["f1"]
    active3 = heldout["active"][3]["f1"]
    query_count = heldout["query_count"]

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    writeup = f"""# MagicCut: uncertainty-aware material grouping with active graph correction

## Abstract

Material Magic Wand retrieves material-consistent components of an untextured, pre-segmented mesh from one query part. We study whether its released 1152-D part representation can support structured binary inference and explicit positive/negative clarification without retraining the encoder. MagicCut calibrates per-part retrieval scores on a mesh-disjoint validation split, builds a k-nearest-neighbour graph over released part features, minimizes a Potts graph-cut energy, and converts uncertain decisions into binary user questions. On the locked 94-mesh held-out split ({query_count} queries), {graph_text_inline} {active_text} The locked experiment therefore rejects both the headline graph-superiority hypothesis and the predefined secondary success criterion. It shows only that feedback repairs part of a substantially degraded graph solution; it does not establish a new state of the art or a human-interaction advantage.

## 1. Problem statement

The input is an untextured mesh already decomposed into fine-grained parts and one selected query part. The output is the subset of parts intended to share a material with that query. This is intra-mesh grouping, not initial part segmentation, material recognition from texture, or cross-object retrieval.

Material-aware grouping is intrinsically ambiguous: geometry, context, and artistic intent can disagree. A single global distance threshold cannot express “this similar-looking piece is not part of my intended group.” Our research question is therefore not only whether a graph improves a frozen retrieval decision, but whether calibrated uncertainty and explicit positive/negative constraints reduce correction effort.

## 2. Material Magic Wand

Material Magic Wand renders every part in three configurations: isolated part, part with context, and full object. A DINO-v3-small encoder processes each image; concatenation yields the 1152-D pre-projection feature used for retrieval. Training uses a 128-D projection and supervised contrastive loss to pull parts with the same within-mesh material ID together and push different-material parts apart. At inference, candidates are ranked by L1 distance to the query feature and grouped with a validation-selected tolerance.

The paper defines similarity as negative L1 distance but prints an inequality whose sign conflicts with its tolerance description. We do not silently choose between the two textual conventions: this implementation explicitly uses and tests `L1 distance <= threshold`, which matches the described behaviour that a larger distance tolerance selects more parts. The released deduplication mapping is applied so identical components share one representative embedding.

## 3. Proposed extension

MagicCut leaves the released encoder frozen. A validation-fitted logistic calibrator converts query-candidate evidence into probabilities. For each mesh, a symmetric {GRAPH['k']}-nearest-neighbour graph uses the isolated-part feature family; pairwise affinities use temperature factor {GRAPH['sigma_factor']}. Binary grouping minimizes calibrated unary costs plus a Potts term with weight {GRAPH['pairwise_lambda']}, while the query and later feedback become hard constraints.

After each inference step, an influence-aware acquisition rule combines calibration entropy and diversity from already queried parts (weights {INTERACTION['weights']['uncertainty']}, {INTERACTION['weights']['diversity']}; graph-degree weight {INTERACTION['weights']['degree']}). The selected candidate receives a simulated yes/no answer from benchmark ground truth and graph inference is rerun, for at most {INTERACTION['click_budget']} clicks.

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

{table}

Material Magic Wand F1 with mesh-bootstrap 95% interval: {interval(ci35['material_magic_wand.f1'])}. MagicCut F1: {interval(ci35['magiccut.f1'])}. Three-click active F1: {interval(ci35['active.click_3.f1'])}.

{graph_text} {active_text}

The common frozen embedding ranking has held-out mAP {f(heldout['retrieval']['average_precision'])}, PR AUC {f(heldout['retrieval']['pr_auc'])}, R-precision {f(heldout['retrieval']['r_precision'])}, and Recall@20 {f(heldout['retrieval']['recall_at_20'])}. Graph ranking AP in statistical analysis is explicitly decision-aware (selected partition first with calibrated probability only as a tie-break) and is not described as a native continuous graph score.

## 7. Ablations and gates

- The validation graph sweep covered k in {{5, 10, 20, 50}}, six pairwise weights, three affinity temperatures, and isolated-part/context/combined edges. The frozen choice is the configuration stated above.
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

Neither predefined success criterion is met on the locked held-out evaluation. The evidence should be stated exactly as follows: {graph_text} {active_text} This is a useful negative result: validation gains did not generalize, degradation grows sharply with mesh size, and three oracle-labelled corrections recover only part of the loss. A stronger novelty or effectiveness claim would require an author-verified reproduction, redesigned scalable graph inference, external benchmarks, and a real user study.

## Primary sources

- Jain et al., Material Magic Wand, CVPR 2026: <https://arxiv.org/abs/2603.17370>
- Jain, Mirzaei, and Gilitschenski, GaussianCut, NeurIPS 2024: <https://arxiv.org/abs/2411.07555>
"""
    (docs / "research_writeup.md").write_text(writeup, encoding="utf-8")

    summary = f"""# MagicCut - one-page technical summary

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

{table}

{graph_text} {active_text}

## Honest interpretation

Neither success criterion is met: graph inference is substantially worse than direct retrieval, and three simulated corrections do not recover the baseline. The defensible contribution is a reproducible diagnosis of validation-to-test failure and a measured partial repair from explicit feedback. Answers are simulated, this is one benchmark, and the released authors' preprocessing code is unavailable.
"""
    (docs / "technical_summary.md").write_text(summary, encoding="utf-8")

    demo_script = f"""# Three-minute MagicCut demonstration script

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

"On {query_count} queries from 94 held-out meshes, Material Magic Wand reaches F1 {f(mmw)}, no-click MagicCut reaches {f(cut)}, and three simulated corrections reach {f(active3)}. {graph_text}"

## 2:15-2:40 - Failure case

Open the deterministic failure panel.

"Graph smoothing can connect similar-looking parts that ground truth treats as different materials. This is why the project keeps explicit negative correction and does not claim graph inference always helps."

## 2:40-3:00 - Takeaway

"The locked result is negative: this graph construction does not generalize and three corrections do not recover the direct baseline. The useful outcome is a reproducible failure diagnosis and evidence that feedback repairs part of the graph error. The next step is an author-verified reproduction and a redesigned scalable graph before any artist study."
"""
    (docs / "three_minute_demo_script.md").write_text(demo_script, encoding="utf-8")

    outreach = f"""# Concise outreach package for Umangi Jain

## Suggested subject

Reproducing Material Magic Wand + an active graph-correction experiment

## Draft message

Hi Umangi,

I am an MSc student at U of T interested in TISL's work on interactive 3D perception. I read Material Magic Wand and GaussianCut, then built a reproduction-first experiment that keeps the released Material Magic Wand encoder frozen and tests calibrated graph inference with explicit positive/negative clarification.

I used the released 100-mesh/241-query benchmark, verified every render archive and embedding cache, froze all parameters on a mesh-disjoint validation split, and evaluated the remaining 94 meshes. The honest result is: {graph_text_inline} After three simulated binary corrections, held-out mean F1 is {f(active3)} versus {f(mmw)} for the Material Magic Wand threshold baseline; the paired mesh-bootstrap delta is {interval(active_delta)}.

I also built an interactive demo, failure analysis, and fully locked evaluation. I am not claiming a new state of the art - I would especially value your advice on whether the preprocessing matches your intended release pipeline and whether negative clarification is a useful direction for the project.

Would you be open to a short conversation? I can send a concise technical summary, exact reproduction commands, and the demo.

Best,  
[Name]

## Attach or link

1. One-page technical summary (`docs/technical_summary.md`).
2. Method figure and principal result table.
3. Short demo GIF and three-minute script.
4. Repository only after the numerical audit and clean-environment reproduction pass.

## Conversation points

- Confirm released preprocessing and the intended threshold sign convention.
- Ask whether the original 5-mesh/13-query validation identities can be shared.
- Discuss whether negative evidence matches likely artist workflows.
- Propose a small real-user click study rather than over-interpreting simulated feedback.
"""
    (docs / "outreach_package.md").write_text(outreach, encoding="utf-8")

    readme = f"""# MagicCut

An interactive computer vision project for grouping parts of a 3D object that should share the same material.

![MagicCut interaction](reports/generated/presentation/demo.gif)

## Why this problem matters

A detailed 3D model can contain hundreds or thousands of separate parts. Before adding colour and texture, an artist may need to identify every part that should be made of the same material. Selecting those parts one at a time is slow and mistakes are easy to make.

[Material Magic Wand](https://arxiv.org/abs/2603.17370) approaches this as a visual retrieval problem. The artist clicks one part, the model compares it with every other part in the object, and similar parts are selected automatically.

The difficult cases are not always solved by visual similarity alone. Two parts can look alike but need different materials. A matching set can also contain pieces with different shapes or positions. MagicCut explores whether relationships between parts and a few yes or no corrections can make that grouping process more reliable.

## What I built

I started from the public Material Magic Wand model and benchmark, then built a complete evaluation and interaction pipeline around it:

1. Audited the released model checkpoint, benchmark metadata, and 100 mesh archives.
2. Extracted and verified 25,348 part embeddings from three render contexts per part.
3. Reproduced the paper's distance-based material grouping baseline.
4. Converted similarity scores into calibrated probabilities.
5. Built a nearest-neighbour graph and used graph cut inference to group related parts.
6. Added an active clarification policy that chooses the most useful part to ask about next.
7. Built an interactive demo that updates the group after positive or negative feedback.
8. Froze every parameter before running the final 94-mesh test split.

The project combines computer vision, graph algorithms, uncertainty estimation, active interaction, statistical evaluation, and reproducible ML engineering. The released encoder remains frozen, so the experiment measures the value of the new inference and feedback system rather than additional model training.

## How MagicCut works

![MagicCut method](reports/generated/figures/method_diagram.png)

Each mesh is already divided into parts. The user selects one query part, shown in red in the demo.

1. A frozen DINO-v3-small vision encoder represents each part using isolated, contextual, and full-object renders.
2. A calibration model estimates how likely every candidate is to belong with the query.
3. A graph connects visually related parts and a graph cut produces the initial group.
4. Uncertain parts are shown in yellow.
5. The system recommends one clarification, outlined in blue.
6. A yes or no answer becomes a hard constraint and the graph is solved again.

This interaction can repeat for up to three corrections. Selected parts are orange and rejected parts remain grey.

## Research question

The main question was simple: can graph structure improve one-click material grouping, and can one to three targeted corrections recover difficult cases?

All graph parameters, thresholds, calibration settings, and interaction weights were chosen using six validation meshes. The remaining 94 meshes were held out until the configuration was frozen.

## Results

The final test contains {query_count} grouping queries across 94 held-out meshes.

{table}

The original Material Magic Wand threshold remained the strongest method overall. MagicCut reached F1 {f(cut)} without feedback, compared with {f(mmw)} for the baseline. Three targeted corrections raised MagicCut to {f(active3)}, recovering part of the loss but not reaching the baseline.

The paired difference after three corrections was {interval(active_delta)}. Its 95% mesh-bootstrap interval stayed below zero, so the final experiment does not support a claim that MagicCut outperforms Material Magic Wand.

![F1 versus number of corrections](reports/generated/figures/f1_versus_clicks.png)

The graph performed well on the validation meshes, then became too conservative on larger unseen meshes. I kept and analyzed that regression because it revealed where the approach breaks. The experiment also showed that explicit feedback consistently repaired some graph errors, with a mean F1 gain of {interval(correction_delta)} over zero-click MagicCut.

## Interactive demo

The demo lets a user choose a mesh and query part, inspect the predicted group, answer the recommended yes or no question, and watch the grouping update.

```bash
uv sync --frozen --extra dev
uv run python scripts/serve_demo.py
```

Open `http://127.0.0.1:8000` after the server starts.

The evaluation uses benchmark labels to simulate answers. The interface itself accepts real user input, but this project does not claim results from a human user study.

## Repository guide

| Path | Purpose |
|---|---|
| `src/magiccut/` | Core calibration, graph inference, uncertainty, and feedback code |
| `demo/` | Browser interface for interactive grouping |
| `scripts/` | Data audits, embedding extraction, experiments, figures, and demo server |
| `tests/` | Unit and HTTP integration tests |
| `reports/generated/` | Frozen configurations, metrics, statistical analysis, and figures |
| `docs/research_writeup.md` | Full research report and limitations |
| `docs/technical_summary.md` | One-page technical summary |

## Run the project

Requirements: Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

```bash
uv sync --frozen --extra dev
uv run pytest -q
```

Download and verify the public checkpoint and benchmark metadata:

```bash
uv run python scripts/audit_resources.py --download-checkpoint
uv run python scripts/audit_benchmark.py --download-metadata
uv run python scripts/prepare_benchmark.py
```

Extract embeddings and reproduce the final analysis:

```bash
uv run python scripts/extract_embeddings.py
uv run python scripts/evaluate_locked.py
uv run python scripts/analyze_statistics.py
uv run python scripts/generate_figures.py
uv run python scripts/audit_project.py
```

The evaluation script refuses to overwrite an existing final result. This prevents accidental test-set tuning. Large datasets, checkpoints, and embedding caches are intentionally excluded from Git.

To verify the repository from a clean temporary environment:

```bash
make clean-check
```

The clean check installs only from the lockfile, runs all unit and integration tests, and audits the Git payload for accidental large files.

## Evaluation details

- Dataset: 100 meshes, 241 queries, and 25,348 representative parts
- Split: 6 validation meshes and 94 held-out meshes
- Metrics: macro precision, recall, F1, retrieval mAP, and risk coverage
- Statistics: 5,000 bootstrap samples grouped by mesh
- Reproducibility: archive hashes, feature shapes, IDs, finite values, and cache hashes are checked
- Testing: 24 unit and integration tests, plus a clean-environment installation test

![Held-out precision and recall](reports/generated/figures/precision_recall.png)

## Failure analysis

The best graph-cut improvement and the largest regression are shown below. Both examples were selected automatically from the held-out results using the change in F1, not by manual cherry-picking.

![Largest held-out improvement](reports/generated/figures/success_example.png)

![Largest held-out regression](reports/generated/figures/failure_example.png)

Performance dropped most sharply on large meshes. Graph smoothing sometimes connected visually similar parts that had different material labels, while conservative unary probabilities missed small or disconnected positive groups. Positive and negative feedback helped correct these cases but three answers were not enough to eliminate the gap.

## Limitations

- Feedback labels are simulated from benchmark ground truth, not collected from 3D artists.
- The public release does not include the authors' full inference implementation, so preprocessing was reconstructed and audited from the paper, checkpoint, and released data.
- The six validation meshes used here are not presented as the paper's private validation split.
- The released images provide three contexts from one camera, not independent camera views.
- Results come from one benchmark and do not establish performance on other 3D datasets.

## License

The original MagicCut source code is available under the [MIT License](LICENSE). Material Magic Wand, GaussianCut, the released checkpoint, and the benchmark are third-party resources and remain subject to their original licenses and terms. The checkpoint and full benchmark archives are downloaded separately and are not redistributed in this repository.

## References

- [Material Magic Wand paper](https://arxiv.org/abs/2603.17370), [project page](https://umangi-jain.github.io/material-magic-wand/), [benchmark](https://huggingface.co/datasets/umangijain/material-magic-wand), and [checkpoint](https://huggingface.co/umangijain/material-magic-wand)
- [GaussianCut paper](https://arxiv.org/abs/2411.07555) and [code](https://github.com/umangi-jain/gaussiancut)

For the complete methodology and statistical analysis, see the [research write-up](docs/research_writeup.md).
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    claims = {
        "heldout_query_count": {"value": query_count, "source": "locked_evaluation.report.heldout_94_meshes.query_count"},
        "heldout_mesh_count": {"value": heldout["mesh_count"], "source": "locked_evaluation.report.heldout_94_meshes.mesh_count"},
        "mmw_f1": {"value": mmw, "source": "locked_evaluation.report.heldout_94_meshes.methods.material_magic_wand.f1"},
        "magiccut_f1": {"value": cut, "source": "locked_evaluation.report.heldout_94_meshes.methods.magiccut.f1"},
        "active_3_f1": {"value": active3, "source": "locked_evaluation.report.heldout_94_meshes.active[3].f1"},
        "active_3_delta": {"value": active_delta, "source": "statistics.report.heldout_94_meshes.mesh_bootstrap_95_ci"},
        "active_3_minus_magiccut_delta": {"value": correction_delta, "source": "statistics.report.heldout_94_meshes.mesh_bootstrap_95_ci"},
        "full_query_count": {"value": full["query_count"], "source": "locked_evaluation.report.full_100_meshes.query_count"},
        "raw_results_sha256": {"value": report35["raw_results_sha256"], "source": "sha256(raw_results.json)"},
        "fixed_paper_facts": {
            "value": {"meshes": 100, "queries": 241, "feature_dimension": 1152, "render_contexts": 3},
            "source": "Material Magic Wand arXiv:2603.17370v1 and audited release metadata",
        },
    }
    generated = [ROOT / "README.md", docs / "research_writeup.md", docs / "technical_summary.md", docs / "three_minute_demo_script.md", docs / "outreach_package.md"]
    forbidden = re.compile(r"\b(statistically significant|state[- ]of[- ]the[- ]art|novel)\b", re.I)
    wording_hits = []
    for path in generated:
        for match in forbidden.finditer(path.read_text(encoding="utf-8")):
            wording_hits.append({"path": str(path.relative_to(ROOT)), "term": match.group(0)})
    allowed_context = {"state of the art", "novelty", "novel"}
    unsupported = [hit for hit in wording_hits if hit["term"].lower() not in allowed_context]
    if unsupported:
        raise ValueError(f"Unsupported claim wording: {unsupported}")
    audit = {
        "status": "passed",
        "source_raw_results_sha256": report35["raw_results_sha256"],
        "claims": claims,
        "generated_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in generated],
        "wording_audit_hits_reviewed": wording_hits,
        "policy": "All experiment numbers are formatted from locked evaluation/36 data; fixed benchmark facts are grounded in the primary paper and audited release metadata.",
    }
    output = ROOT / "reports/generated/presentation"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "numerical_claim_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
