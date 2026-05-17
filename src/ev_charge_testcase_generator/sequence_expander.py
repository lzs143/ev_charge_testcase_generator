"""基于交直流时序知识库展开可执行步骤。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sequence_knowledge import SequenceKnowledgeLoader, SequenceKnowledgeNotFoundError
from .semantic_extractor import SemanticExtractionResult


MESSAGE_DEFAULT_PARAMETERS: dict[str, dict[str, str]] = {
    message: {
        "message": message,
        "version": "GBT2023",
        "cycle_ms": "250",
        "expected_cycle_ms": "250",
        "message_id": "default",
        "expected_message_id": "default",
        "message_variant": "valid",
    }
    for message in ("CHM", "CRM", "CTS", "CML", "CRO", "CCS", "CST", "CSD", "CEM")
}

FAULT_VARIANT_BY_TYPE: dict[str, str] = {
    "报文内容错误": "invalid_content",
    "报文周期错误": "invalid_cycle",
    "报文ID错误": "invalid_id",
}


@dataclass
class SequenceExpansionResult:
    """时序知识库展开结果。"""

    matched: bool
    steps: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    cleanup_steps: list[dict[str, Any]]
    metadata: dict[str, Any]


class SequenceExpander:
    """将匹配到的充电时序知识展开为动作、判据和恢复步骤。"""

    def __init__(self, sequence_loader: SequenceKnowledgeLoader | None = None) -> None:
        """初始化时序展开器。"""

        self.sequence_loader = sequence_loader or SequenceKnowledgeLoader()

    def expand(self, semantic_result: SemanticExtractionResult) -> SequenceExpansionResult:
        """根据语义抽取结果展开完整动作级测试步骤。"""

        if not semantic_result.protocol_flow:
            return self._unmatched("未识别主时序模板")

        try:
            knowledge = self.sequence_loader.get_by_flow_id(semantic_result.protocol_flow)
            stage = self.sequence_loader.find_stage(
                semantic_result.protocol_flow,
                semantic_result.test_stage,
                semantic_result.message_types,
            )
            interaction = self.sequence_loader.find_interaction(
                semantic_result.protocol_flow,
                semantic_result.test_stage,
                semantic_result.message_types,
                semantic_result.fault_type,
            )
        except (FileNotFoundError, SequenceKnowledgeNotFoundError) as exc:
            return self._unmatched(str(exc))

        if stage is None or interaction is None:
            return self._unmatched("未匹配到可展开的阶段或交互")

        step_items: list[dict[str, Any]] = []
        step_items.extend(knowledge.global_entry_steps)
        for section_name in ("stimulus", "response"):
            section = interaction.get(section_name)
            if isinstance(section, dict):
                step_items.append(section)
        step_items.extend(interaction.get("exit_steps", []))

        metadata = {
            "matched": True,
            "flow_id": knowledge.flow_id,
            "flow_name": knowledge.flow_name,
            "standard_basis": knowledge.standard_basis,
            "default_parameters": knowledge.default_parameters,
            "matched_stage_id": stage.get("stage_id"),
            "matched_stage_name": stage.get("stage_name"),
            "matched_interaction_id": interaction.get("interaction_id"),
            "matched_interaction_name": interaction.get("name"),
            "expanded_step_count": len(step_items),
            "expanded_assertion_count": len(interaction.get("checks", [])),
            "expanded_cleanup_step_count": len(knowledge.global_cleanup_steps),
        }
        return SequenceExpansionResult(
            matched=True,
            steps=self._build_steps(step_items, semantic_result),
            assertions=self._build_assertions(interaction.get("checks", []), semantic_result),
            cleanup_steps=self._build_cleanup_steps(knowledge.global_cleanup_steps),
            metadata=metadata,
        )

    @staticmethod
    def _build_steps(items: list[dict[str, Any]], semantic_result: SemanticExtractionResult) -> list[dict[str, Any]]:
        """将知识库动作定义转换为 executable_info.steps。"""

        steps: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            steps.append(
                {
                    "step_id": index,
                    "action_id": item.get("action_id"),
                    "action_name": str(item.get("action_name") or item.get("name") or item.get("description") or ""),
                    "action_type": str(item.get("action_type") or "执行动作"),
                    "target": item.get("target") or item.get("receiver") or semantic_result.target_object,
                    "parameters": SequenceExpander._build_step_parameters(item, semantic_result),
                    "message": item.get("message"),
                    "signal": item.get("signal"),
                    "duration_ms": item.get("duration_ms"),
                    "timeout_ms": item.get("timeout_ms") or item.get("monitor_duration_ms"),
                    "required": bool(item.get("required", True)),
                    "description": str(item.get("description") or item.get("action_name") or ""),
                }
            )
        return steps

    @staticmethod
    def _build_step_parameters(item: dict[str, Any], semantic_result: SemanticExtractionResult) -> dict[str, str]:
        """合并基础动作默认参数、知识库参数和语义故障注入参数。"""

        message = str(item.get("message") or "").upper()
        parameters: dict[str, str] = {}
        is_send_action = SequenceExpander._is_send_action(item)
        if is_send_action and message in MESSAGE_DEFAULT_PARAMETERS:
            parameters.update(MESSAGE_DEFAULT_PARAMETERS[message])

        for key in ("message", "version", "cycle_ms", "expected_cycle_ms", "message_id", "expected_message_id"):
            if item.get(key) is not None:
                parameters[key] = str(item[key])

        parameters.update({key: str(value) for key, value in dict(item.get("parameters", {})).items()})
        if is_send_action:
            parameters.update({key: str(value) for key, value in semantic_result.parameters.items()})
            if semantic_result.fault_type is not None:
                parameters["fault_type"] = semantic_result.fault_type
                parameters.setdefault("message_variant", FAULT_VARIANT_BY_TYPE.get(semantic_result.fault_type, "invalid"))
            if semantic_result.fault_type == "报文内容错误":
                parameters.setdefault("content_error_type", "data_content_error")
            if parameters.get("content_error_type") and parameters.get("message_variant") == "valid":
                parameters["message_variant"] = "invalid_content"
            if parameters.get("cycle_ms") != parameters.get("expected_cycle_ms") and semantic_result.fault_type == "报文周期错误":
                parameters["message_variant"] = "invalid_cycle"
            if parameters.get("message_id") != parameters.get("expected_message_id") and semantic_result.fault_type == "报文ID错误":
                parameters["message_variant"] = "invalid_id"

        return {key: value for key, value in parameters.items() if value != ""}

    @staticmethod
    def _is_send_action(item: dict[str, Any]) -> bool:
        """判断知识库条目是否属于主动发送报文的基础动作。"""

        action_id = str(item.get("action_id") or "")
        action_name = str(item.get("action_name") or item.get("name") or item.get("description") or "")
        action_type = str(item.get("action_type") or "")
        return action_id.startswith("SEND_") or "发送" in action_name or "发送" in action_type

    @staticmethod
    def _build_assertions(items: list[dict[str, Any]], semantic_result: SemanticExtractionResult) -> list[dict[str, Any]]:
        """将知识库判据定义转换为 executable_info.assertions。"""

        assertions: list[dict[str, Any]] = []
        for item in items:
            assertions.append(
                {
                    "assertion_id": item.get("assertion_id"),
                    "assertion_type": str(item.get("assertion_type") or "state"),
                    "description": str(item.get("description") or item.get("expected_value") or ""),
                    "target": item.get("target") or semantic_result.target_object,
                    "signal": item.get("signal"),
                    "message": item.get("message"),
                    "operator": item.get("operator"),
                    "expected_value": item.get("expected_value"),
                    "timeout_ms": item.get("timeout_ms"),
                }
            )
        return assertions

    @staticmethod
    def _build_cleanup_steps(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将知识库恢复动作转换为 executable_info.cleanup_steps。"""

        cleanup_steps: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            cleanup_steps.append(
                {
                    "step_id": index,
                    "action_id": item.get("action_id"),
                    "action_name": str(item.get("action_name") or item.get("description") or ""),
                    "parameters": dict(item.get("parameters", {})),
                    "required": bool(item.get("required", True)),
                }
            )
        return cleanup_steps

    @staticmethod
    def _unmatched(reason: str) -> SequenceExpansionResult:
        """构造未匹配结果，调用方可回退到粗粒度规则。"""

        return SequenceExpansionResult(
            matched=False,
            steps=[],
            assertions=[],
            cleanup_steps=[],
            metadata={"matched": False, "reason": reason},
        )
