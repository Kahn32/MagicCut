import json
from pathlib import Path
import numpy as np

REPORT=Path("reports/generated/interaction_experiments/report.json")

def load_report():
    assert REPORT.exists(),"Run scripts/run_interaction_experiments.py first"
    return json.loads(REPORT.read_text())

def test_protocol_and_regressions_are_frozen():
    report=load_report(); protocol=report["protocol"]
    assert set(protocol["validation_uids"]).isdisjoint(protocol["test_uids"])
    assert protocol["cache_hits"]==10 and protocol["test_not_used_for_threshold_or_policy_selection"]
    assert np.isclose(report["prompt_21"]["test"]["calibrated"]["f1"],.7510416666666666)
    assert np.isclose(report["prompt_21"]["test"]["magiccut"]["f1"],.7253151260504203)
    assert report["prompt_21"]["graph_gate"]["passed"]

def test_failure_and_min_marginal_audits_are_complete():
    report=load_report()
    assert len(report["prompt_22"]["validation"]["all"])==11
    assert len(report["prompt_22"]["test"]["all"])==8
    assert report["prompt_23"]["validation"]["cut_solves"]==397
    assert report["prompt_23"]["test"]["cut_solves"]==358
    assert all(row["gap"]>=0 for split in ("validation_most_ambiguous","test_most_ambiguous")
               for row in report["prompt_23"][split])

def test_every_feedback_history_respects_oracle_constraints():
    report=load_report()
    results=[]
    for number in (26,27,28):
        results.extend(report[f"prompt_{number}"][split] for split in ("validation","test"))
    for mode in ("positive_only","negative_only","unrestricted_binary"):
        results.extend(report["prompt_29"][mode][split] for split in ("validation","test"))
    for result in results:
            for history in result["histories"]:
                for row in history:
                    assert set(row["positive"])<=set(row["selected"])
                    assert set(row["negative"]).isdisjoint(row["selected"])
                    assert row["click"] in range(4)

def test_policy_was_selected_on_validation_and_then_passed_claim_gate():
    report=load_report(); prompt=report["prompt_30"]
    assert prompt["validation_policy_selection"]=="influence_aware"
    assert prompt["claim_gate"]["passed_gate_3"]
    for curves in (prompt["all_validation_curves"],prompt["all_test_curves"]):
        assert all([row["click"] for row in curve]==[0,1,2,3] for curve in curves.values())
