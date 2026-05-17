from __future__ import annotations

from ev_charge_testcase_generator.sequence_expander import SequenceExpander
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractionResult, SemanticExtractor


def test_sequence_expander_expands_dc_chm_bhm_steps() -> None:
    semantic_result = SemanticExtractor().extract("测试直流充电过程中，BMS是否能正常回复BHM报文。")

    expansion = SequenceExpander().expand(semantic_result)

    assert expansion.matched is True
    assert expansion.metadata["matched_interaction_id"] == "DC_CHM_BHM_HANDSHAKE"
    assert [step["action_id"] for step in expansion.steps] == [
        "PHY_LOCK-00001",
        "CAN_CONFIG-00002",
        "AUX_POWER_ON-00003",
        "SLEEP-2000",
        "CAN_RECORD_START-00004",
        "SEND_CHM-00005",
        "WAIT_BHM-00006",
        "STOP_CHM-00009",
    ]
    assert [assertion["assertion_id"] for assertion in expansion.assertions] == [
        "CHECK_BHM_PERIOD-00007",
        "CHECK_BHM_CONTENT-00008",
    ]
    assert expansion.cleanup_steps[-1]["action_id"] == "CAN_RECORD_STOP-00023"


def test_sequence_expander_applies_default_parameters_to_other_charger_messages() -> None:
    semantic_result = SemanticExtractionResult(
        is_valid=True,
        raw_text="",
        normalized_text="",
        message="",
        scene_type="DC",
        condition_type="normal",
        test_stage="dc_parameter_config",
        target_object="BMS/EVCC",
        protocol_flow="DC_CHARGING_SEQUENCE",
        message_types=["CML", "BCP"],
    )

    expansion = SequenceExpander().expand(semantic_result)

    send_step = next(step for step in expansion.steps if step["action_id"] == "SEND_CML-00013")
    assert send_step["parameters"]["message"] == "CML"
    assert send_step["parameters"]["message_variant"] == "valid"
    assert send_step["parameters"]["message_id"] == "default"
    assert send_step["parameters"]["cycle_ms"] == "250"


def test_sequence_expander_prefers_invalid_chm_bem_interaction() -> None:
    semantic_result = SemanticExtractor().extract("发送错误的CHM报文，检查BMS是否正常回复BEM报文")

    expansion = SequenceExpander().expand(semantic_result)

    assert expansion.matched is True
    assert expansion.metadata["matched_interaction_id"] == "DC_INVALID_CHM_BEM_ERROR_RESPONSE"
    assert expansion.steps[5]["message"] == "CHM"
    assert expansion.steps[5]["parameters"]["message_variant"] == "invalid_content"
    assert expansion.steps[5]["parameters"]["cycle_ms"] == "250"
    assert expansion.steps[5]["parameters"]["message_id"] == "default"
    assert expansion.steps[6]["message"] == "BEM"
    assert expansion.assertions[0]["message"] == "BEM"


def test_sequence_expander_applies_chm_cycle_error_parameters() -> None:
    semantic_result = SemanticExtractor().extract("发送周期错误的CHM报文，周期为500ms，检查BMS回复BEM报文")

    expansion = SequenceExpander().expand(semantic_result)

    assert semantic_result.fault_type == "报文周期错误"
    assert expansion.matched is True
    assert expansion.metadata["matched_interaction_id"] == "DC_CHM_CYCLE_ERROR_BEM_RESPONSE"
    assert expansion.steps[5]["parameters"]["message_variant"] == "invalid_cycle"
    assert expansion.steps[5]["parameters"]["cycle_ms"] == "500"
    assert expansion.steps[5]["parameters"]["expected_cycle_ms"] == "250"


def test_sequence_expander_applies_chm_id_error_parameters() -> None:
    semantic_result = SemanticExtractor().extract("发送报文ID错误的CHM报文，报文ID为0x123，检查BMS回复BEM报文")

    expansion = SequenceExpander().expand(semantic_result)

    assert semantic_result.fault_type == "报文ID错误"
    assert expansion.matched is True
    assert expansion.metadata["matched_interaction_id"] == "DC_CHM_ID_ERROR_BEM_RESPONSE"
    assert expansion.steps[5]["parameters"]["message_variant"] == "invalid_id"
    assert expansion.steps[5]["parameters"]["message_id"] == "0x123"
    assert expansion.steps[5]["parameters"]["expected_message_id"] == "default"


def test_sequence_expander_distinguishes_chm_timeout_from_invalid_chm() -> None:
    semantic_result = SemanticExtractor().extract("充电握手阶段，测试系统未收到CHM报文且等待超时后，EVCC应发送BEM错误报文。")

    expansion = SequenceExpander().expand(semantic_result)

    assert semantic_result.fault_type == "报文超时"
    assert expansion.matched is True
    assert expansion.metadata["matched_interaction_id"] == "DC_CHM_TIMEOUT_BEM_ERROR_RESPONSE"
    assert expansion.steps[5]["message"] == "CHM"
    assert expansion.steps[5]["timeout_ms"] == 70000
    assert expansion.steps[6]["message"] == "BEM"
