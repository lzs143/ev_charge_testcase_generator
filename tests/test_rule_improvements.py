from __future__ import annotations

from ev_charge_testcase_generator.action_mapper import ActionMapper
from ev_charge_testcase_generator.assertion_mapper import AssertionMapper
from ev_charge_testcase_generator.extractor import RuleBasedExtractor
from ev_charge_testcase_generator.models import Assertion, ExecutableStep
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractor


def test_extractor_extracts_multiple_message_expected_results() -> None:
    text = "充电阶段，EVCC周期发送BCL电池充电需求报文、BCS电池充电总状态报文和BSM车辆状态信息报文。"

    result = RuleBasedExtractor().extract(text)

    assert result.scene_type == "DC"
    assert "发送BCL电池充电需求报文" in result.expected_results
    assert "发送BCS电池充电总状态报文" in result.expected_results
    assert "发送BSM车辆状态信息报文" in result.expected_results


def test_extractor_does_not_treat_allowed_voltage_as_allow_charge() -> None:
    text = "直流充电参数配置阶段，设置最高允许充电电压为750V，最大允许充电电流为250A。"

    result = RuleBasedExtractor().extract(text)

    assert result.parameters == {"voltage": "750V", "current": "250A"}
    assert "完成参数配置" in result.expected_results
    assert "允许充电" not in result.expected_results


def test_extractor_filters_noise_text() -> None:
    result = RuleBasedExtractor().extract("今天学习了BMS通信协议的基本概念，不生成测试用例。")

    assert result.scene_type is None
    assert "无法判断充电场景类型" in result.warnings


def test_semantic_stage_prefers_parameter_config_for_parameter_messages() -> None:
    text = "当IUT收到的BCP、CML或BCL报文电流值超出规定范围时，应退出充电过程。"

    result = SemanticExtractor().extract(text)

    assert result.is_valid is True
    assert result.test_stage == "充电参数配置阶段"
    assert result.fault_type == "电流越界"


def test_action_mapper_maps_abstract_fault_and_energy_transfer_steps() -> None:
    mapper = ActionMapper()
    fault_step = ExecutableStep(
        step_id=1,
        action_id=None,
        action_name="CP信号异常",
        action_type="执行动作",
        target="CP",
        parameters={},
        message=None,
        signal="CP",
        duration_ms=None,
        timeout_ms=None,
        description="交流充电过程中，当CP信号异常时，应停止充电。",
    )
    energy_step = ExecutableStep(
        step_id=2,
        action_id=None,
        action_name="进入能量传输阶段",
        action_type="执行动作",
        target="BMS/EVCC",
        parameters={},
        message=None,
        signal=None,
        duration_ms=None,
        timeout_ms=None,
        description="系统进入能量传输阶段。",
    )

    assert mapper.map_step(fault_step).action_id == "AC-DC_STOP-00246"
    assert mapper.map_step(energy_step).action_id == "AC_START-00020"


def test_assertion_mapper_maps_extended_message_and_state_assertions() -> None:
    mapper = AssertionMapper()
    message_assertion = Assertion(
        assertion_id=None,
        assertion_type="message",
        description="发送BCS电池充电总状态报文",
        target="EVCC",
        message="BCS",
        expected_value="发送BCS电池充电总状态报文",
    )
    state_assertion = Assertion(
        assertion_id=None,
        assertion_type="state",
        description="记录故障信息",
        target="BMS/EVCC",
        expected_value="记录故障信息",
    )

    assert mapper.map_assertion(message_assertion).assertion_id == "PJ-BCS-001"
    assert mapper.map_assertion(state_assertion).assertion_id == "PJ-FAULT-RECORD"
