"""可执行测试用例生成模块。"""

from __future__ import annotations

from typing import Any

from .action_mapper import ActionMapper
from .assertion_mapper import AssertionMapper
from .models import Assertion, CleanupStep, ExecutableStep, ExecutableTestCase, Precondition


class ExecutableTestCaseGenerator:
    """根据可执行语义标注生成自动化测试用例。"""

    def __init__(
        self,
        action_mapper: ActionMapper | None = None,
        assertion_mapper: AssertionMapper | None = None,
    ) -> None:
        """初始化生成器。"""

        self.action_mapper = action_mapper or ActionMapper()
        self.assertion_mapper = assertion_mapper or AssertionMapper()

    def generate(self, executable_info: dict[str, Any]) -> ExecutableTestCase | None:
        """从 executable_gold 风格的结构化信息生成可执行测试用例。"""

        if not executable_info.get("is_relevant", True):
            return None

        preconditions = [
            self._build_precondition(item)
            for item in executable_info.get("preconditions", [])
        ]
        steps = [
            self.action_mapper.map_step(self._build_step(item))
            for item in executable_info.get("steps", [])
        ]
        assertions = [
            self.assertion_mapper.map_assertion(self._build_assertion(item))
            for item in executable_info.get("assertions", [])
        ]
        cleanup_steps = [
            self.action_mapper.map_cleanup_step(self._build_cleanup_step(item))
            for item in executable_info.get("cleanup_steps", [])
        ]

        return ExecutableTestCase(
            case_id=str(executable_info["case_id"]),
            case_name=str(executable_info["case_name"]),
            scene_type=str(executable_info["scene_type"]),
            condition_type=str(executable_info["condition_type"]),
            test_type=str(executable_info["test_type"]),
            standard_source=executable_info.get("standard_source"),
            test_stage=executable_info.get("test_stage"),
            target_object=executable_info.get("target_object"),
            tester_role=executable_info.get("tester_role"),
            protocol_flow=executable_info.get("protocol_flow"),
            preconditions=preconditions,
            steps=steps,
            assertions=assertions,
            cleanup_steps=cleanup_steps,
            parameters=dict(executable_info.get("parameters", {})),
            fault_type=executable_info.get("fault_type"),
            raw_requirement=str(executable_info.get("raw_requirement", "")),
            metadata=dict(executable_info.get("metadata", {})),
        )

    @staticmethod
    def _build_precondition(data: dict[str, Any]) -> Precondition:
        """构造前置条件。"""

        return Precondition(
            condition_id=str(data["condition_id"]),
            description=str(data["description"]),
            target=data.get("target"),
            parameters=dict(data.get("parameters", {})),
            required=bool(data.get("required", True)),
        )

    @staticmethod
    def _build_step(data: dict[str, Any]) -> ExecutableStep:
        """构造可执行动作步骤。"""

        return ExecutableStep(
            step_id=int(data["step_id"]),
            action_id=data.get("action_id"),
            action_name=str(data["action_name"]),
            action_type=str(data["action_type"]),
            target=data.get("target"),
            parameters=dict(data.get("parameters", {})),
            message=data.get("message"),
            signal=data.get("signal"),
            duration_ms=data.get("duration_ms"),
            timeout_ms=data.get("timeout_ms"),
            required=bool(data.get("required", True)),
            description=str(data.get("description", "")),
        )

    @staticmethod
    def _build_assertion(data: dict[str, Any]) -> Assertion:
        """构造自动化判据。"""

        return Assertion(
            assertion_id=data.get("assertion_id"),
            assertion_type=str(data["assertion_type"]),
            description=str(data["description"]),
            target=data.get("target"),
            signal=data.get("signal"),
            message=data.get("message"),
            operator=data.get("operator"),
            expected_value=data.get("expected_value"),
            timeout_ms=data.get("timeout_ms"),
        )

    @staticmethod
    def _build_cleanup_step(data: dict[str, Any]) -> CleanupStep:
        """构造恢复步骤。"""

        return CleanupStep(
            step_id=int(data["step_id"]),
            action_id=data.get("action_id"),
            action_name=str(data["action_name"]),
            parameters=dict(data.get("parameters", {})),
            required=bool(data.get("required", True)),
        )
