"""交直流充电主时序知识库加载与匹配模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SequenceKnowledgeNotFoundError(KeyError):
    """时序知识不存在时抛出的明确异常。"""


@dataclass
class SequenceKnowledge:
    """交流或直流充电主时序知识。"""

    flow_id: str
    flow_name: str
    scene_type: str
    target_object: str
    tester_role: str
    standard_basis: list[str]
    default_parameters: dict[str, Any]
    global_entry_steps: list[dict[str, Any]]
    stages: list[dict[str, Any]]
    global_cleanup_steps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，便于 GUI 或生成器直接展示。"""

        return {
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "scene_type": self.scene_type,
            "target_object": self.target_object,
            "tester_role": self.tester_role,
            "standard_basis": self.standard_basis,
            "default_parameters": self.default_parameters,
            "global_entry_steps": self.global_entry_steps,
            "stages": self.stages,
            "global_cleanup_steps": self.global_cleanup_steps,
        }


class SequenceKnowledgeLoader:
    """从 JSON 文件读取交直流充电主时序知识。"""

    DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "sequence_knowledge"

    def __init__(self, knowledge_dir: Path | str | None = None) -> None:
        """初始化知识库目录。"""

        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir is not None else self.DEFAULT_DIR

    def load_all(self) -> list[SequenceKnowledge]:
        """读取目录下的所有时序知识文件。"""

        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"时序知识库目录不存在: {self.knowledge_dir}")

        return [self._load_file(path) for path in sorted(self.knowledge_dir.glob("*.json"))]

    def get_by_flow_id(self, flow_id: str) -> SequenceKnowledge:
        """根据主时序标识返回知识对象。"""

        for knowledge in self.load_all():
            if knowledge.flow_id == flow_id:
                return knowledge
        raise SequenceKnowledgeNotFoundError(f"未找到时序知识: {flow_id}")

    def find_stage(self, flow_id: str, test_stage: str | None, message_types: list[str]) -> dict[str, Any] | None:
        """根据测试阶段或报文类型匹配最相关的流程阶段。"""

        knowledge = self.get_by_flow_id(flow_id)
        normalized_stage = test_stage or ""
        message_set = {message.upper() for message in message_types}
        for stage in knowledge.stages:
            stage_name = str(stage.get("stage_name", ""))
            keywords = [str(keyword) for keyword in stage.get("keywords", [])]
            if normalized_stage and (normalized_stage == stage_name or normalized_stage in stage_name):
                return stage
            if any(keyword.upper() in message_set for keyword in keywords):
                return stage
            if normalized_stage and any(keyword in normalized_stage for keyword in keywords):
                return stage
        return None

    def find_interaction(
        self,
        flow_id: str,
        test_stage: str | None,
        message_types: list[str],
        fault_type: str | None = None,
    ) -> dict[str, Any] | None:
        """根据阶段和报文类型匹配典型交互。"""

        stage = self.find_stage(flow_id, test_stage, message_types)
        if stage is None:
            return None

        interactions = stage.get("interactions", [])
        message_set = {message.upper() for message in message_types}
        if message_set:
            scored_interactions: list[tuple[int, dict[str, Any]]] = []
            for interaction in interactions:
                interaction_messages = self._collect_interaction_messages(interaction)
                score = len(message_set & interaction_messages)
                interaction_fault_type = interaction.get("fault_type")
                if interaction_fault_type and fault_type == interaction_fault_type:
                    score += 5
                elif interaction_fault_type and fault_type != interaction_fault_type:
                    score -= 100
                if score > 0:
                    scored_interactions.append((score, interaction))
            if scored_interactions:
                return max(scored_interactions, key=lambda item: item[0])[1]
        return interactions[0] if interactions else None

    @staticmethod
    def _collect_interaction_messages(interaction: dict[str, Any]) -> set[str]:
        """收集交互定义中涉及的报文名称。"""

        messages: set[str] = set()
        for section_name in ("stimulus", "response"):
            section = interaction.get(section_name)
            if not isinstance(section, dict):
                continue
            message = section.get("message")
            if isinstance(message, str):
                messages.update(item.strip().upper() for item in message.replace("，", ",").split(",") if item.strip())
        for check in interaction.get("checks", []):
            message = check.get("message")
            if isinstance(message, str):
                messages.add(message.upper())
        return messages

    @staticmethod
    def _load_file(path: Path) -> SequenceKnowledge:
        """读取单个时序知识 JSON 文件。"""

        with path.open("r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        return SequenceKnowledge(
            flow_id=str(data["flow_id"]),
            flow_name=str(data["flow_name"]),
            scene_type=str(data["scene_type"]),
            target_object=str(data["target_object"]),
            tester_role=str(data["tester_role"]),
            standard_basis=list(data.get("standard_basis", [])),
            default_parameters=dict(data.get("default_parameters", {})),
            global_entry_steps=list(data.get("global_entry_steps", [])),
            stages=list(data.get("stages", [])),
            global_cleanup_steps=list(data.get("global_cleanup_steps", [])),
        )
