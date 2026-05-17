"""GB/T 27930 报文字段知识库加载与查询。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MessageFieldKnowledgeNotFoundError(KeyError):
    """报文或字段知识不存在时抛出的明确异常。"""


@dataclass
class MessageFieldDefinition:
    """单个报文字段定义。"""

    field_id: str
    field_name: str
    spn: str | None = None
    aliases: list[str] = field(default_factory=list)
    semantic_keys: list[str] = field(default_factory=list)
    value_type: str = "string"
    unit: str | None = None
    required: bool = False
    default_value: str | None = None
    normal_values: list[str] = field(default_factory=list)
    invalid_values: list[str] = field(default_factory=list)
    valid_range: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageFieldDefinition":
        """从 JSON 字典构造字段定义。"""

        return cls(
            field_id=str(data["field_id"]),
            field_name=str(data["field_name"]),
            spn=data.get("spn"),
            aliases=[str(value) for value in data.get("aliases", [])],
            semantic_keys=[str(value) for value in data.get("semantic_keys", [])],
            value_type=str(data.get("value_type", "string")),
            unit=data.get("unit"),
            required=bool(data.get("required", False)),
            default_value=data.get("default_value"),
            normal_values=[str(value) for value in data.get("normal_values", [])],
            invalid_values=[str(value) for value in data.get("invalid_values", [])],
            valid_range=data.get("valid_range"),
        )

    def matches(self, query: str) -> bool:
        """判断自然语言字段名或语义键是否命中该字段。"""

        normalized_query = query.strip().upper()
        if not normalized_query:
            return False
        candidates = [self.field_id, self.field_name, *(self.spn or "",), *self.aliases, *self.semantic_keys]
        return any(normalized_query == candidate.upper() for candidate in candidates if candidate)


@dataclass
class MessageDefinition:
    """单个报文的字段知识定义。"""

    message: str
    message_name: str
    sender: str
    receiver: str
    stage: str
    default_cycle_ms: int
    fields: list[MessageFieldDefinition]

    @classmethod
    def from_dict(cls, message: str, data: dict[str, Any]) -> "MessageDefinition":
        """从 JSON 字典构造报文定义。"""

        return cls(
            message=message,
            message_name=str(data.get("message_name", "")),
            sender=str(data.get("sender", "")),
            receiver=str(data.get("receiver", "")),
            stage=str(data.get("stage", "")),
            default_cycle_ms=int(data.get("default_cycle_ms", 250)),
            fields=[MessageFieldDefinition.from_dict(item) for item in data.get("fields", [])],
        )

    def find_field(self, query: str) -> MessageFieldDefinition | None:
        """在该报文内按字段名、别名、SPN 或语义键查询字段。"""

        for field_definition in self.fields:
            if field_definition.matches(query):
                return field_definition
        return None


class MessageFieldKnowledgeLoader:
    """读取和查询 GB/T 27930 报文字段知识库。"""

    DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "message_field_knowledge" / "dc_message_fields.json"

    def __init__(self, knowledge_path: Path | str | None = None) -> None:
        """初始化知识库文件路径。"""

        self.knowledge_path = Path(knowledge_path) if knowledge_path is not None else self.DEFAULT_PATH

    def load(self) -> dict[str, Any]:
        """读取原始知识库 JSON。"""

        if not self.knowledge_path.exists():
            raise FileNotFoundError(f"报文字段知识库不存在: {self.knowledge_path}")
        return json.loads(self.knowledge_path.read_text(encoding="utf-8"))

    def load_messages(self) -> dict[str, MessageDefinition]:
        """读取所有报文字段定义并按报文名索引。"""

        data = self.load()
        return {
            message.upper(): MessageDefinition.from_dict(message.upper(), definition)
            for message, definition in data.get("messages", {}).items()
        }

    def get_message(self, message: str) -> MessageDefinition:
        """按报文名查询字段定义。"""

        messages = self.load_messages()
        message_key = message.upper()
        if message_key not in messages:
            raise MessageFieldKnowledgeNotFoundError(f"未找到报文字段定义: {message}")
        return messages[message_key]

    def find_field(self, message: str, query: str) -> MessageFieldDefinition | None:
        """按报文名和自然语言字段描述查询字段。"""

        return self.get_message(message).find_field(query)

