from __future__ import annotations

import pytest

from ev_charge_testcase_generator.pipeline import TestCaseGenerationPipeline


def _pipeline() -> TestCaseGenerationPipeline:
    return TestCaseGenerationPipeline()


def test_pipeline_generates_dc_executable_parameter_case() -> None:
    result = _pipeline().run("车辆完成握手后，设置直流充电电压为500V，电流为30A。")

    assert result.extracted_info.scene_type == "DC"
    assert result.executable_info["parameters"] == {"voltage": "500V", "current": "30A"}
    assert result.test_case.scene_type == "DC"
    assert result.test_case.condition_type == "normal"
    assert result.test_case.parameters == {"voltage": "500V", "current": "30A"}
    assert result.test_case.steps[0].action_id == "ASET-00001-0"
    assert result.check_result.passed is True


def test_pipeline_generates_dc_bms_fault_executable_case() -> None:
    result = _pipeline().run("直流充电过程中，当BMS通信中断时，系统应停止输出并记录故障信息。")

    assert result.extracted_info.fault_type == "BMS通信中断"
    assert result.test_case.condition_type == "fault"
    assert result.test_case.test_type == "negative"
    assert result.test_case.fault_type == "BMS通信中断"
    assert any(assertion.assertion_id == "PJ-STOP-OUTPUT" for assertion in result.test_case.assertions)
    assert result.check_result.passed is True


def test_pipeline_generates_invalid_chm_bem_fault_case() -> None:
    result = _pipeline().run("发送错误的CHM报文，检查BMS是否正常回复BEM报文")

    assert result.test_case.condition_type == "fault"
    assert result.test_case.test_type == "negative"
    assert result.test_case.fault_type == "报文内容错误"
    assert result.executable_info["metadata"]["sequence_knowledge"]["matched_interaction_id"] == (
        "DC_INVALID_CHM_BEM_ERROR_RESPONSE"
    )
    assert result.test_case.steps[5].message == "CHM"
    assert result.test_case.steps[5].parameters["message_variant"] == "invalid_content"
    assert result.test_case.steps[5].parameters["cycle_ms"] == "250"
    assert result.test_case.steps[5].parameters["message_id"] == "default"
    assert result.test_case.steps[6].message == "BEM"
    assert result.test_case.assertions[0].assertion_id == "PJ-BEM-001"
    assert result.check_result.passed is True


def test_pipeline_generates_chm_cycle_error_fault_case() -> None:
    result = _pipeline().run("发送周期错误的CHM报文，周期为500ms，检查BMS回复BEM报文")

    assert result.test_case.fault_type == "报文周期错误"
    assert result.executable_info["metadata"]["sequence_knowledge"]["matched_interaction_id"] == (
        "DC_CHM_CYCLE_ERROR_BEM_RESPONSE"
    )
    assert result.test_case.steps[5].parameters["message_variant"] == "invalid_cycle"
    assert result.test_case.steps[5].parameters["cycle_ms"] == "500"
    assert result.test_case.steps[5].parameters["expected_cycle_ms"] == "250"
    assert result.check_result.passed is True


def test_pipeline_generates_wrong_cycle_word_order_chm_fault_case() -> None:
    result = _pipeline().run("发送错误周期CHM报文，检查BMS能否正确回复BEM报文")

    assert result.test_case.fault_type == "报文周期错误"
    assert result.executable_info["metadata"]["sequence_knowledge"]["matched_interaction_id"] == (
        "DC_CHM_CYCLE_ERROR_BEM_RESPONSE"
    )
    assert result.test_case.steps[5].parameters["message_variant"] == "invalid_cycle"
    assert result.test_case.steps[5].parameters["cycle_ms"] == "500"
    assert result.test_case.steps[5].parameters["expected_cycle_ms"] == "250"
    assert result.test_case.steps[6].message == "BEM"
    assert result.check_result.passed is True


def test_pipeline_generates_chm_id_error_fault_case() -> None:
    result = _pipeline().run("发送报文ID错误的CHM报文，报文ID为0x123，检查BMS回复BEM报文")

    assert result.test_case.fault_type == "报文ID错误"
    assert result.executable_info["metadata"]["sequence_knowledge"]["matched_interaction_id"] == (
        "DC_CHM_ID_ERROR_BEM_RESPONSE"
    )
    assert result.test_case.steps[5].parameters["message_variant"] == "invalid_id"
    assert result.test_case.steps[5].parameters["message_id"] == "0x123"
    assert result.test_case.steps[5].parameters["expected_message_id"] == "default"
    assert result.check_result.passed is True


def test_pipeline_generates_ac_cp_fault_executable_case() -> None:
    result = _pipeline().run("交流充电过程中，当CP信号异常时，应停止充电并提示异常状态。")

    assert result.extracted_info.scene_type == "AC"
    assert result.extracted_info.fault_type == "CP信号异常"
    assert result.test_case.scene_type == "AC"
    assert result.test_case.test_stage == "导引控制阶段"
    assert any(step.signal == "CP" for step in result.test_case.steps)
    assert any(assertion.assertion_id == "PJ-STOP-CHARGE" for assertion in result.test_case.assertions)
    assert result.check_result.passed is True


def test_pipeline_rejects_irrelevant_text() -> None:
    with pytest.raises(ValueError):
        _pipeline().run("请整理交流会议纪要，并统计参会人员名单。")


def test_pipeline_run_batch() -> None:
    results = _pipeline().run_batch(
        [
            "车辆完成握手后，设置直流充电电压为500V，电流为30A。",
            "交流充电过程中，当CP信号异常时，应停止充电并提示异常状态。",
        ]
    )

    assert len(results) == 2
    assert [result.test_case.scene_type for result in results] == ["DC", "AC"]
