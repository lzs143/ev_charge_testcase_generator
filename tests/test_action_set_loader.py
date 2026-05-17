from __future__ import annotations

import pytest

from ev_charge_testcase_generator.action_set_loader import ActionSetLoader, ActionSetNotFoundError


def test_action_set_loader_loads_dc_actions() -> None:
    loader = ActionSetLoader("data/action_sets/dc_action_set.json")

    actions = loader.load_all()

    assert len(actions) >= 30
    assert loader.get_by_id("SEND_CHM-00005").action_name == "周期发送CHM报文"
    assert "message_variant" in loader.get_by_id("SEND_CHM-00005").parameter_names
    assert "expected_message_id" in loader.get_by_id("SEND_CHM-00005").parameter_names
    assert loader.get_by_id("CHECK_BHM_PERIOD-00007").default_values == "BHM,250"


def test_charger_sent_message_actions_use_fault_injection_parameters() -> None:
    loader = ActionSetLoader("data/action_sets/dc_action_set.json")
    expected_action_ids = [
        "SEND_CHM-00005",
        "SEND_CRM-00010",
        "SEND_CML-00013",
        "SEND_CRO-00015",
        "SEND_CTS-00026",
        "SEND_CCS-00027",
        "SEND_CST-00028",
        "SEND_CSD-00029",
        "SEND_CEM-00030",
    ]

    for action_id in expected_action_ids:
        action = loader.get_by_id(action_id)

        assert action.parameter_names == (
            "message,version,cycle_ms,message_id,message_variant,fault_type,"
            "content_error_type,field_name,field_value,expected_cycle_ms,expected_message_id"
        )
        assert action.default_values.split(",")[4] == "valid"
        assert action.default_values.split(",")[9:] == ["250", "default"]


def test_action_set_loader_raises_clear_error_when_missing() -> None:
    loader = ActionSetLoader("data/action_sets/dc_action_set.json")

    with pytest.raises(ActionSetNotFoundError, match="未找到动作编号: missing"):
        loader.get_by_id("missing")
