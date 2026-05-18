from __future__ import annotations

from ev_charge_testcase_generator.semantic_events import build_semantic_events
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractionResult
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractor


def test_semantic_events_split_stimulus_and_expected_response() -> None:
    result = SemanticExtractor().extract("发送CHM报文，查看BMS是否正确回复BHM报文")

    events = build_semantic_events(result)

    assert events.protocol_messages == ["CHM", "BHM"]
    assert events.communication_objects == ["BMS"]
    assert events.trigger_condition == "发送CHM报文后"
    assert [event.message for event in events.stimulus] == ["CHM"]
    assert [event.message for event in events.expected_response] == ["BHM"]
    assert [event.message for event in events.checks] == ["BHM"]
    assert events.expected_response[0].actor == "BMS"


def test_semantic_events_do_not_treat_bms_as_protocol_message() -> None:
    result = SemanticExtractor().extract("发送CHM报文，查看BMS是否正确回复BHM报文")

    events = build_semantic_events(result)

    assert "BMS" not in events.protocol_messages


def test_semantic_events_capture_state_feedback_rules() -> None:
    result = SemanticExtractionResult(
        is_valid=True,
        raw_text="CP信号异常时，系统应停止充电并提示异常状态",
        normalized_text="CP信号异常时，系统应停止充电并提示异常状态",
        message="ok",
        scene_type="AC",
        condition_type="fault",
        test_type="negative",
        target_object="BMS/EVCC",
        tester_role="测试系统/交流供电设备",
    )

    events = build_semantic_events(result)

    assert [event.action for event in events.expected_response] == ["stop_charging", "show_warning"]
    assert [event.action for event in events.checks] == ["check_stop_charging", "check_show_warning"]


def test_semantic_events_capture_parameter_setting_and_state_transition() -> None:
    result = SemanticExtractionResult(
        is_valid=True,
        raw_text="设置输出电压为900V，检查系统是否进入异常处理流程",
        normalized_text="设置输出电压为900V，检查系统是否进入异常处理流程",
        message="ok",
        scene_type="DC",
        condition_type="fault",
        test_type="negative",
        target_object="BMS/EVCC",
        tester_role="测试系统/SECC",
    )

    events = build_semantic_events(result)

    assert any(event.action == "set_parameter" for event in events.stimulus)
    assert any(event.action == "enter_state" for event in events.expected_response)
