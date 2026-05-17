from __future__ import annotations

from ev_charge_testcase_generator.semantic_extractor import SemanticExtractor


def test_semantic_extractor_accepts_ac_cp_fault_requirement() -> None:
    result = SemanticExtractor().extract("交流充电过程中，当CP信号异常时，应停止充电并提示异常状态。")

    assert result.is_valid is True
    assert result.scene_type == "AC"
    assert result.condition_type == "fault"
    assert result.test_stage == "导引控制阶段"
    assert result.target_object == "BMS/EVCC"
    assert result.tester_role == "测试系统/交流供电设备"
    assert result.protocol_flow == "AC_CHARGING_SEQUENCE"
    assert result.fault_type == "CP信号异常"
    assert result.signals == ["CP"]
    assert "停止充电" in result.expected_results


def test_semantic_extractor_accepts_dc_message_timeout_requirement() -> None:
    result = SemanticExtractor().extract("充电握手阶段，测试系统未收到CHM报文且等待超时后，EVCC应发送BEM错误报文。")

    assert result.is_valid is True
    assert result.scene_type == "DC"
    assert result.condition_type == "fault"
    assert result.test_stage == "低压辅助上电及充电握手阶段"
    assert result.target_object == "BMS/EVCC"
    assert result.tester_role == "测试系统/SECC"
    assert result.protocol_flow == "DC_CHARGING_SEQUENCE"
    assert result.message_types == ["CHM", "BEM"]
    assert result.standard_source == "GB/T 34658-2025"


def test_semantic_extractor_accepts_invalid_chm_bem_fault_requirement() -> None:
    result = SemanticExtractor().extract("发送错误的CHM报文，检查BMS是否正常回复BEM报文")

    assert result.is_valid is True
    assert result.scene_type == "DC"
    assert result.condition_type == "fault"
    assert result.test_type == "negative"
    assert result.fault_type == "报文内容错误"
    assert result.test_stage == "低压辅助上电及充电握手阶段"
    assert result.message_types == ["CHM", "BEM"]


def test_semantic_extractor_accepts_wrong_cycle_chm_bem_fault_requirement() -> None:
    result = SemanticExtractor().extract("发送错误周期CHM报文，检查BMS能否正确回复BEM报文")

    assert result.is_valid is True
    assert result.condition_type == "fault"
    assert result.fault_type == "报文周期错误"
    assert result.message_types == ["CHM", "BEM"]
    assert result.expected_results == ["发送BEM报文"]


def test_semantic_extractor_rejects_irrelevant_input() -> None:
    result = SemanticExtractor().extract("请整理交流会议纪要，并统计参会人员名单。")

    assert result.is_valid is False
    assert result.message == "请输入正确的充电测试语句。"
    assert len(result.examples) == 3
