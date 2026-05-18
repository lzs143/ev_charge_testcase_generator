"""面向可执行测试用例生成的信息抽取模块。"""

from __future__ import annotations

import re
from typing import Any

from .models import ExtractedInfo
from .sequence_expander import SequenceExpander
from .sequence_knowledge import SequenceKnowledgeLoader, SequenceKnowledgeNotFoundError
from .semantic_events import build_semantic_events
from .semantic_extractor import SemanticExtractionResult, SemanticExtractor


class RuleBasedExecutableExtractor:
    """将基础抽取结果扩展为可执行测试用例语义结构。"""

    MESSAGE_PATTERN = re.compile(
        r"(?:EVCC|SECC|BMS|BHM|BRM|BCP|BCL|BCS|BSM|BEM|CHM|CRM|CML|BRO|CRO|CCS|CEM|BST|CST|BSD|CSD)"
    )

    STAGE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("低压辅助上电及充电握手阶段", ("低压辅助", "握手", "BHM", "CHM")),
        ("辨识阶段", ("辨识", "BRM", "CRM")),
        ("充电参数配置阶段", ("参数配置", "BCP", "CML")),
        ("充电准备就绪阶段", ("准备就绪", "BRO", "CRO")),
        ("充电阶段", ("充电阶段", "BCL", "BCS", "BSM", "CCS", "能量传输")),
        ("充电结束阶段", ("充电结束", "BSD", "CSD")),
        ("充电中止阶段", ("中止", "BST", "CST", "停机")),
        ("导引控制阶段", ("导引", "CP")),
        ("连接确认阶段", ("连接确认", "CC", "CC2")),
    )

    def __init__(
        self,
        semantic_extractor: SemanticExtractor | None = None,
        sequence_loader: SequenceKnowledgeLoader | None = None,
    ) -> None:
        """初始化可执行语义构建器。"""

        self.semantic_extractor = semantic_extractor or SemanticExtractor()
        self.sequence_loader = sequence_loader or SequenceKnowledgeLoader()
        self.sequence_expander = SequenceExpander(self.sequence_loader)

    def extract(self, text: str, extracted_info: ExtractedInfo) -> dict[str, Any]:
        """生成 executable_info，作为 ExecutableTestCaseGenerator 的输入。"""

        semantic_result = self.semantic_extractor.extract(text)
        if not semantic_result.is_valid:
            return {
                "is_relevant": False,
                "reject_reason": semantic_result.message,
                "examples": semantic_result.examples,
                "warnings": semantic_result.warnings,
            }

        scene_type = str(semantic_result.scene_type)
        condition_type = semantic_result.condition_type or "normal"
        messages = semantic_result.message_types
        semantic_events = build_semantic_events(semantic_result)
        target = semantic_result.target_object or self._infer_target_object(text, scene_type)
        test_stage = semantic_result.test_stage
        test_type = semantic_result.test_type or ("negative" if condition_type == "fault" else "positive")
        sequence_expansion = self.sequence_expander.expand(semantic_result)
        steps = (
            sequence_expansion.steps
            if sequence_expansion.matched
            else self._build_steps(text, scene_type, target, messages, extracted_info.parameters)
        )
        assertions = (
            sequence_expansion.assertions
            if sequence_expansion.matched
            else self._build_assertions(semantic_result.expected_results, target, messages)
        )
        cleanup_steps = (
            sequence_expansion.cleanup_steps
            if sequence_expansion.matched
            else self._build_cleanup_steps(scene_type)
        )

        return {
            "is_relevant": True,
            "case_id": self._build_case_id(scene_type, condition_type, text),
            "case_name": text.rstrip("。"),
            "scene_type": scene_type,
            "condition_type": condition_type,
            "test_type": test_type,
            "standard_source": semantic_result.standard_source,
            "test_stage": test_stage,
            "target_object": target,
            "tester_role": semantic_result.tester_role,
            "protocol_flow": semantic_result.protocol_flow,
            "message_types": messages,
            "preconditions": self._build_preconditions(scene_type, target, test_stage),
            "steps": steps,
            "assertions": assertions,
            "cleanup_steps": cleanup_steps,
            "parameters": semantic_result.parameters,
            "fault_type": semantic_result.fault_type,
            "raw_requirement": text,
            "metadata": {
                "extractor": "rule_based_executable",
                "semantic": self._semantic_metadata(semantic_result),
                "semantic_events": semantic_events.to_dict(),
                "sequence_knowledge": sequence_expansion.metadata,
                "sequence_expanded": sequence_expansion.matched,
            },
        }

    @staticmethod
    def _semantic_metadata(semantic_result: SemanticExtractionResult) -> dict[str, Any]:
        """将语义抽取结果中的辅助信息写入元数据。"""

        return {
            "trigger_condition": semantic_result.trigger_condition,
            "signals": semantic_result.signals,
            "actions": semantic_result.actions,
            "tester_role": semantic_result.tester_role,
            "protocol_flow": semantic_result.protocol_flow,
            "confidence": semantic_result.confidence,
            "warnings": semantic_result.warnings,
        }

    def _sequence_metadata(self, semantic_result: SemanticExtractionResult) -> dict[str, Any]:
        """根据主时序模板匹配充电时序知识。"""

        if not semantic_result.protocol_flow:
            return {"matched": False, "reason": "未识别主时序模板"}

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
            return {"matched": False, "reason": str(exc)}

        return {
            "matched": True,
            "flow_id": knowledge.flow_id,
            "flow_name": knowledge.flow_name,
            "standard_basis": knowledge.standard_basis,
            "default_parameters": knowledge.default_parameters,
            "matched_stage_id": stage.get("stage_id") if stage else None,
            "matched_stage_name": stage.get("stage_name") if stage else None,
            "matched_interaction_id": interaction.get("interaction_id") if interaction else None,
            "matched_interaction_name": interaction.get("name") if interaction else None,
            "global_entry_step_count": len(knowledge.global_entry_steps),
            "global_cleanup_step_count": len(knowledge.global_cleanup_steps),
        }

    def _extract_messages(self, text: str, expected_results: list[str]) -> list[str]:
        """抽取报文和通信对象缩略语。"""

        values = self.MESSAGE_PATTERN.findall(text)
        for expected_result in expected_results:
            values.extend(self.MESSAGE_PATTERN.findall(expected_result))
        return self._unique(values)

    @staticmethod
    def _infer_target_object(text: str, scene_type: str) -> str:
        """推断被测对象。"""

        if "EVCC" in text:
            return "EVCC"
        if "SECC" in text:
            return "SECC"
        if "BMS" in text:
            return "BMS"
        if scene_type == "AC":
            return "车辆" if "车辆" in text else "供电设备"
        return "充电系统"

    def _infer_test_stage(self, text: str) -> str | None:
        """根据阶段关键词推断测试阶段。"""

        upper_text = text.upper()
        for stage, keywords in self.STAGE_RULES:
            if any(keyword.upper() in upper_text for keyword in keywords):
                return stage
        return None

    @staticmethod
    def _infer_standard_source(scene_type: str, messages: list[str]) -> str | None:
        """根据场景和报文推断标准来源。"""

        if scene_type == "DC" and messages:
            return "GB/T 34658-2025"
        if scene_type == "DC":
            return "GB/T 27930-2023"
        if scene_type == "AC":
            return "GB/T 18487.1-2023"
        return None

    @staticmethod
    def _build_preconditions(scene_type: str, target: str, test_stage: str | None) -> list[dict[str, Any]]:
        """生成默认前置条件。"""

        descriptions = (
            ["测试系统和被测对象完成物理连接", "直流充电通信配置完成"]
            if scene_type == "DC"
            else ["交流充电接口连接准备完成", "供电设备和车辆连接状态可检测"]
        )
        if test_stage:
            descriptions.append(f"进入{test_stage}")
        return [
            {
                "condition_id": f"PRE-{index:03d}",
                "description": description,
                "target": target,
                "parameters": {},
                "required": True,
            }
            for index, description in enumerate(descriptions, start=1)
        ]

    def _build_steps(
        self,
        text: str,
        scene_type: str,
        target: str,
        messages: list[str],
        parameters: dict[str, str],
    ) -> list[dict[str, Any]]:
        """生成默认自动化动作步骤。"""

        steps: list[dict[str, Any]] = [
            {
                "step_id": 1,
                "action_id": None,
                "action_name": "直流插枪初始化" if scene_type == "DC" else "交流插枪初始化",
                "action_type": "初始化",
                "target": target,
                "parameters": {},
                "message": None,
                "signal": None,
                "duration_ms": None,
                "timeout_ms": None,
                "required": True,
                "description": "建立测试初始状态",
            },
            {
                "step_id": 2,
                "action_id": None,
                "action_name": "直流插枪" if scene_type == "DC" else "交流插枪",
                "action_type": "连接控制",
                "target": target,
                "parameters": {},
                "message": None,
                "signal": "CC2" if scene_type == "DC" else "CP/CC",
                "duration_ms": None,
                "timeout_ms": None,
                "required": True,
                "description": "建立充电连接",
            },
        ]

        action_description = self._infer_action_description(text, messages, parameters)
        is_timeout = any(keyword in text for keyword in ("超时", "未收到", "收不到"))
        steps.append(
            {
                "step_id": 3,
                "action_id": None,
                "action_name": action_description[:40],
                "action_type": self._infer_action_type(action_description),
                "target": target,
                "parameters": parameters,
                "message": ",".join(messages) if messages else None,
                "signal": self._infer_signal(text),
                "duration_ms": 70000 if is_timeout else None,
                "timeout_ms": 70000 if is_timeout else None,
                "required": True,
                "description": text,
            }
        )
        return steps

    @staticmethod
    def _infer_action_description(text: str, messages: list[str], parameters: dict[str, str]) -> str:
        """推断核心测试动作描述。"""

        if parameters:
            return "设置测试参数"
        if messages and not any(keyword in text for keyword in ("未收到", "不发送", "收不到")):
            return "按测试需求发送或等待相关报文"
        return text

    @staticmethod
    def _infer_action_type(description: str) -> str:
        """推断动作类型。"""

        if "等待" in description or "超时" in description:
            return "等待"
        if "发送" in description:
            return "发送报文"
        if any(keyword in description for keyword in ("未收到", "不发送", "停发", "收不到")):
            return "故障注入"
        if "设置" in description or "配置" in description:
            return "设置参数"
        if any(keyword in description for keyword in ("停止", "退出", "中止")):
            return "状态控制"
        if "插枪" in description or "连接" in description:
            return "连接控制"
        return "执行动作"

    @staticmethod
    def _infer_signal(text: str) -> str | None:
        """推断涉及的物理信号。"""

        if "CP" in text:
            return "CP"
        if "CC2" in text:
            return "CC2"
        if "CC" in text:
            return "CC"
        return None

    def _build_assertions(
        self,
        expected_results: list[str],
        target: str,
        messages: list[str],
    ) -> list[dict[str, Any]]:
        """生成自动化判据语义。"""

        assertions: list[dict[str, Any]] = []
        for expected_result in expected_results:
            expected_messages = self._unique(self.MESSAGE_PATTERN.findall(expected_result))
            assertion_type = "message" if expected_messages or "报文" in expected_result else "state"
            assertion_messages = expected_messages or messages
            assertions.append(
                {
                    "assertion_id": None,
                    "assertion_type": assertion_type,
                    "description": expected_result,
                    "target": target,
                    "signal": None,
                    "message": ",".join(assertion_messages) if assertion_type == "message" and assertion_messages else None,
                    "operator": "should_send" if "发送" in expected_result else "should_enter" if "进入" in expected_result else "should_equal",
                    "expected_value": expected_result,
                    "timeout_ms": 70000 if "超时" in expected_result else None,
                }
            )
        return assertions

    @staticmethod
    def _build_cleanup_steps(scene_type: str) -> list[dict[str, Any]]:
        """生成恢复步骤。"""

        names = ["清空消息", "直流高压复位", "直流低压复位"] if scene_type == "DC" else ["停止充电", "交流拔枪恢复"]
        return [
            {
                "step_id": index,
                "action_id": None,
                "action_name": name,
                "parameters": {},
                "required": True,
            }
            for index, name in enumerate(names, start=1)
        ]

    @staticmethod
    def _build_case_id(scene_type: str, condition_type: str, text: str) -> str:
        """生成稳定的可执行用例编号。"""

        return f"TC-{scene_type}-{condition_type.upper()}-{abs(hash(text)) % 1_000_000:06d}"

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """保持顺序去重。"""

        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result
