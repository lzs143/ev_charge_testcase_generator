"""自动化动作集加载模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ActionSetNotFoundError(KeyError):
    """动作集或动作编号不存在时抛出的明确异常。"""


@dataclass
class ActionDefinition:
    """动作集中的单条动作定义。"""

    action_id: str
    action_name: str
    action_type: str
    scene_type: str
    stage: str
    target: str
    object_type: str
    parameter_names: str
    default_values: str
    expected_effect: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """转换为与 Excel 表头一致的字典。"""

        return {
            "动作编号": self.action_id,
            "动作名称": self.action_name,
            "动作类型": self.action_type,
            "适用场景": self.scene_type,
            "适用阶段": self.stage,
            "控制对象": self.target,
            "动作对象类型": self.object_type,
            "参数项": self.parameter_names,
            "默认参数值": self.default_values,
            "预期效果": self.expected_effect,
            "说明": self.description,
        }


class ActionSetLoader:
    """读取 JSON 动作集，供生成器和 GUI 查询动作定义。"""

    DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "action_sets" / "dc_action_set.json"

    def __init__(self, action_set_path: Path | str | None = None) -> None:
        """初始化动作集文件路径。"""

        self.action_set_path = Path(action_set_path) if action_set_path is not None else self.DEFAULT_PATH

    def load_all(self) -> list[ActionDefinition]:
        """读取所有动作定义。"""

        if not self.action_set_path.exists():
            raise FileNotFoundError(f"动作集文件不存在: {self.action_set_path}")

        with self.action_set_path.open("r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        return [self._build_action(item) for item in data.get("actions", [])]

    def get_by_id(self, action_id: str) -> ActionDefinition:
        """根据动作编号查询动作定义。"""

        for action in self.load_all():
            if action.action_id == action_id:
                return action
        raise ActionSetNotFoundError(f"未找到动作编号: {action_id}")

    def to_index(self) -> dict[str, ActionDefinition]:
        """按动作编号构建索引。"""

        return {action.action_id: action for action in self.load_all()}

    @staticmethod
    def _build_action(data: dict[str, Any]) -> ActionDefinition:
        """将与 Excel 表头一致的字典转换为动作定义。"""

        return ActionDefinition(
            action_id=str(data["动作编号"]),
            action_name=str(data.get("动作名称", "")),
            action_type=str(data.get("动作类型", "")),
            scene_type=str(data.get("适用场景", "")),
            stage=str(data.get("适用阶段", "")),
            target=str(data.get("控制对象", "")),
            object_type=str(data.get("动作对象类型", "")),
            parameter_names=str(data.get("参数项", "")),
            default_values=str(data.get("默认参数值", "")),
            expected_effect=str(data.get("预期效果", "")),
            description=str(data.get("说明", "")),
        )
