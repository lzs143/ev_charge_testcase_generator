from __future__ import annotations

from ev_charge_testcase_generator.semantic_events import build_semantic_events
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
