from __future__ import annotations

import json

from ev_charge_testcase_generator.models import (
    Assertion,
    CheckResult,
    CleanupStep,
    ExecutableStep,
    ExecutableTestCase,
    ExtractedInfo,
    Precondition,
    Requirement,
)


def test_requirement_to_dict() -> None:
    requirement = Requirement(
        requirement_id="req_001",
        text="测试交流充电过程",
        metadata={"scene": "AC"},
    )

    result = requirement.to_dict()

    assert result["requirement_id"] == "req_001"
    assert result["text"] == "测试交流充电过程"
    assert result["source"] == "manual"
    assert result["metadata"] == {"scene": "AC"}


def test_extracted_info_to_dict() -> None:
    extracted_info = ExtractedInfo(
        scene_type="DC",
        condition_type="fault",
        objects=["BMS"],
        actions=["停止充电"],
        parameters={"voltage": "750V"},
        trigger_condition="通信中断时",
        fault_type="通信中断",
        expected_results=["停止充电"],
        confidence=0.9,
        warnings=["示例警告"],
    )

    result = extracted_info.to_dict()

    assert result["scene_type"] == "DC"
    assert result["condition_type"] == "fault"
    assert result["parameters"] == {"voltage": "750V"}
    assert result["confidence"] == 0.9


def test_check_result_to_dict() -> None:
    check_result = CheckResult(passed=False, errors=["缺少步骤"], warnings=["存在未映射动作"])

    result = check_result.to_dict()

    assert result["passed"] is False
    assert result["errors"] == ["缺少步骤"]
    assert result["warnings"] == ["存在未映射动作"]
    assert result["fixed_case"] is None


def test_executable_test_case_to_dict_and_to_json() -> None:
    executable_case = ExecutableTestCase(
        case_id="TC-DC-000001",
        case_name="直流充电握手阶段BHM响应测试",
        scene_type="DC",
        condition_type="normal",
        test_type="positive",
        standard_source="GB/T 34658-2025",
        test_stage="低压辅助上电及充电握手阶段",
        target_object="BMS/EVCC",
        preconditions=[
            Precondition(
                condition_id="PRE-001",
                description="测试系统与被测对象完成物理连接",
                target="BMS/EVCC",
            )
        ],
        steps=[
            ExecutableStep(
                step_id=1,
                action_id="PHY_LOCK-00001",
                action_name="直流接口物理锁止",
                action_type="连接控制",
            ),
            ExecutableStep(
                step_id=2,
                action_id="WAIT_BHM-00006",
                action_name="等待BHM报文",
                action_type="等待报文",
                message="BHM",
                timeout_ms=5000,
            ),
        ],
        assertions=[
            Assertion(
                assertion_id="CHECK_BHM_CONTENT-00008",
                assertion_type="message",
                description="检查BHM报文内容",
                target="BMS/EVCC",
                message="BHM",
                operator="should_equal",
            )
        ],
        cleanup_steps=[
            CleanupStep(step_id=1, action_id="CAN_RECORD_STOP-00023", action_name="停止CAN记录")
        ],
        raw_requirement="测试直流充电过程中BMS是否回复BHM报文",
    )

    result = executable_case.to_dict()
    json_result = json.loads(executable_case.to_json())

    assert result["case_id"] == "TC-DC-000001"
    assert result["preconditions"][0]["condition_id"] == "PRE-001"
    assert result["steps"][1]["message"] == "BHM"
    assert result["assertions"][0]["assertion_id"] == "CHECK_BHM_CONTENT-00008"
    assert json_result["cleanup_steps"][0]["action_name"] == "停止CAN记录"
