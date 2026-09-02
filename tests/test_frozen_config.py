from magiccut.frozen import GRAPH, INTERACTION, locked_config


def test_locked_configuration_has_no_test_time_tuning():
    config = locked_config()
    assert config["test_time_tuning"] is False
    assert config["graph"] == GRAPH
    assert config["interaction"] == INTERACTION
    assert len(config["validation_uids"]) == 6


def test_final_interaction_policy_is_influence_aware():
    assert INTERACTION["clarification_policy"] == "influence_aware"
    assert INTERACTION["uncertainty_method"] == "calibration_entropy"
    assert INTERACTION["click_budget"] == 3
