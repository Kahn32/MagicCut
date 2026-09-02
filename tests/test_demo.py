import numpy as np

from magiccut.demo import MagicCutDemo
from magiccut.feedback import make_influence_chooser
from magiccut.frozen import INTERACTION


class StubCalibrator:
    def predict_proba(self, features):
        probability = np.linspace(0.15, 0.85, len(features), dtype=float)
        return np.column_stack([1 - probability, probability])


def make_engine():
    engine = MagicCutDemo.__new__(MagicCutDemo)
    base = np.zeros(1152, dtype=np.float32)
    engine.embeddings = {
        "mesh": {
            1: base.copy(),
            2: np.full(1152, 0.01, dtype=np.float32),
            3: np.full(1152, 0.2, dtype=np.float32),
        }
    }
    engine.calibrator = StubCalibrator()
    engine.sessions = {}
    engine.chooser = make_influence_chooser(
        tuple(INTERACTION["weights"][name] for name in ("uncertainty", "degree", "diversity")),
        INTERACTION["uncertainty_method"],
    )
    return engine


def test_demo_start_feedback_and_reset():
    engine = make_engine()
    result = engine.start("mesh", 1)
    assert result["query_part"] == 1
    assert result["click_count"] == 0
    assert result["recommended_part"] in {2, 3}
    assert set(result["confident_positive"] + result["uncertain"] + result["confident_negative"]) == {1, 2, 3}

    corrected = engine.feedback(result["session_id"], result["recommended_part"], False)
    assert corrected["click_count"] == 1
    assert result["recommended_part"] in corrected["negative_constraints"]

    reset = engine.reset(result["session_id"])
    assert reset["click_count"] == 0
    assert reset["positive_constraints"] == [1]
    assert reset["negative_constraints"] == []
