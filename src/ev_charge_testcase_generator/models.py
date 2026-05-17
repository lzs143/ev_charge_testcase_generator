"""核心数据结构。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Requirement:
    """自然语言测试需求。"""

    requirement_id: str
    text: str
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)


@dataclass
class ExtractedInfo:
    """规则抽取得到的基础语义信息。"""

    scene_type: str | None = None
    condition_type: str | None = None
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    trigger_condition: str | None = None
    fault_type: str | None = None
    expected_results: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)


@dataclass
class Precondition:
    """可执行测试用例的前置条件。"""

    condition_id: str
    description: str
    target: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)


@dataclass
class ExecutableStep:
    """面向自动化平台的动作级测试步骤。"""

    step_id: int
    action_id: str | None
    action_name: str
    action_type: str
    target: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    signal: str | None = None
    duration_ms: int | None = None
    timeout_ms: int | None = None
    required: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)


@dataclass
class Assertion:
    """动作执行后的自动化判据。"""

    assertion_id: str | None
    assertion_type: str
    description: str
    target: str | None = None
    signal: str | None = None
    message: str | None = None
    operator: str | None = None
    expected_value: str | None = None
    timeout_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)


@dataclass
class CleanupStep:
    """测试结束后的恢复动作。"""

    step_id: int
    action_id: str | None
    action_name: str
    parameters: dict[str, str] = field(default_factory=dict)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)


@dataclass
class ExecutableTestCase:
    """动作级可执行测试用例。"""

    case_id: str
    case_name: str
    scene_type: str
    condition_type: str
    test_type: str
    standard_source: str | None
    test_stage: str | None
    target_object: str | None
    tester_role: str | None = None
    protocol_flow: str | None = None
    preconditions: list[Precondition] = field(default_factory=list)
    steps: list[ExecutableStep] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    cleanup_steps: list[CleanupStep] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    fault_type: str | None = None
    raw_requirement: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串。"""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class CheckResult:
    """可执行测试用例检查结果。"""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixed_case: ExecutableTestCase | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典。"""

        return asdict(self)
