from scripts.generate_writeup import conclusions


def test_negative_active_interval_is_not_described_as_including_zero():
    summary = {
        "methods": {
            "material_magic_wand": {"f1": 0.70},
            "magiccut": {"f1": 0.40},
        },
        "active": [{}, {}, {}, {"f1": 0.60}],
    }
    active_vs_mmw = {"estimate": -0.10, "ci_95_low": -0.15, "ci_95_high": -0.05}
    active_vs_cut = {"estimate": 0.20, "ci_95_low": 0.15, "ci_95_high": 0.25}
    graph, active = conclusions(summary, active_vs_mmw, active_vs_cut)
    assert "does not improve" in graph
    assert "below zero" in active
    assert "includes zero" not in active
    assert "graph-induced loss" in active
