"""语义抽取数据集 JSONL 校验脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "semantic_dataset_seed.jsonl"
DEFAULT_MESSAGE_KNOWLEDGE = PROJECT_ROOT / "data" / "message_field_knowledge" / "dc_message_fields.json"

ENTITY_LABELS = {
    "MESSAGE",
    "SIGNAL",
    "OBJECT",
    "INTERFACE",
    "COMPONENT",
    "TEST_LAYER",
    "ACTION",
    "FAULT_EXPR",
    "PARAM_NAME",
    "PARAM_VALUE",
    "FIELD_NAME",
    "EXPECTED_EXPR",
    "TEST_CONDITION",
}

STANDARD_SOURCES = {
    "GB/T 34658-2025",
    "GB/T 34657.2-2017",
    "GB/T 27930-2023",
    "GB/T 18487.1-2023",
}
SCENE_TYPES = {"AC", "DC"}
SYSTEM_CLASSES = {"A", "B", "C", "unknown"}
TEST_LAYERS = {"physical_layer", "data_link_layer", "application_layer", "interoperability"}
CONDITION_TYPES = {"normal", "fault"}
TEST_TYPES = {"positive", "negative"}
TEST_STAGES = {
    "测试准备阶段",
    "低压辅助上电及充电握手阶段",
    "辨识阶段",
    "时间同步阶段",
    "充电参数配置阶段",
    "充电准备就绪阶段",
    "充电阶段",
    "充电结束或中止阶段",
    "错误处理阶段",
    "直流接口连接确认阶段",
    "直流控制导引阶段",
    "交流连接确认阶段",
    "交流控制导引阶段",
    "充电与行驶互锁阶段",
}
FAULT_TYPES = {
    None,
    "报文内容错误",
    "报文周期错误",
    "报文ID错误",
    "报文长度错误",
    "报文格式错误",
    "报文顺序错误",
    "报文超时",
    "通信中断",
    "错误报文响应",
    "状态机跳转错误",
    "绝缘故障",
    "保护接地故障",
    "S+断路",
    "S-断路",
    "S+S-短路",
    "K5状态异常",
    "K6状态异常",
    "低压辅助电源异常",
    "电子锁异常",
    "车辆接口断开",
    "充电与行驶互锁异常",
    "车辆中止充电",
    "充电机中止充电",
    "CP信号异常",
    "CC信号异常",
    "CC2信号异常",
    "CP断路",
    "CC断路",
    "PE断路",
    "控制导引边界异常",
    "电压越界",
    "电流越界",
    "过压",
    "欠压",
    "过流",
    "温度异常",
    "SOC异常",
    "急停",
}
ACTION_INTENTS = {
    "发送报文",
    "等待报文",
    "检查响应",
    "设置参数",
    "故障注入",
    "断开连接",
    "闭合连接",
    "模拟边界",
    "停止输出",
    "记录故障",
    "恢复环境",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 数据集。"""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
    return rows


def load_message_names(path: Path) -> set[str]:
    """读取报文字段知识库中的报文名。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    return {message.upper() for message in data.get("messages", {})}


def validate_dataset(dataset_path: Path, message_knowledge_path: Path) -> list[str]:
    """返回数据集中的所有错误信息。"""

    errors: list[str] = []
    rows = load_jsonl(dataset_path)
    message_names = load_message_names(message_knowledge_path)
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("id", f"line-{index}"))
        text = str(row.get("text", ""))
        if sample_id in seen_ids:
            errors.append(f"{sample_id}: id 重复")
        seen_ids.add(sample_id)
        if not text:
            errors.append(f"{sample_id}: text 不能为空")

        entity_messages: list[str] = []
        for entity in row.get("entities", []):
            label = entity.get("label")
            start = entity.get("start")
            end = entity.get("end")
            entity_text = str(entity.get("text", ""))
            if label not in ENTITY_LABELS:
                errors.append(f"{sample_id}: 未知实体标签 {label}")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                errors.append(f"{sample_id}: 实体 {entity_text} 的 start/end 不合法")
                continue
            if text[start:end] != entity_text:
                errors.append(f"{sample_id}: 实体偏移不匹配 {entity_text!r} != {text[start:end]!r}")
            if label == "MESSAGE":
                entity_messages.append(entity_text.upper())
                if entity_text.upper() not in message_names:
                    errors.append(f"{sample_id}: MESSAGE {entity_text} 不在报文字段知识库中")

        labels = row.get("labels", {})
        _check_value(errors, sample_id, "standard_source", labels.get("standard_source"), STANDARD_SOURCES)
        _check_value(errors, sample_id, "scene_type", labels.get("scene_type"), SCENE_TYPES)
        _check_value(errors, sample_id, "system_class", labels.get("system_class"), SYSTEM_CLASSES)
        _check_value(errors, sample_id, "test_layer", labels.get("test_layer"), TEST_LAYERS)
        _check_value(errors, sample_id, "condition_type", labels.get("condition_type"), CONDITION_TYPES)
        _check_value(errors, sample_id, "test_type", labels.get("test_type"), TEST_TYPES)
        _check_value(errors, sample_id, "test_stage", labels.get("test_stage"), TEST_STAGES)
        _check_value(errors, sample_id, "fault_type", labels.get("fault_type"), FAULT_TYPES)
        for intent in labels.get("action_intent", []):
            if intent not in ACTION_INTENTS:
                errors.append(f"{sample_id}: 未知 action_intent {intent}")

        expected_messages = [message.upper() for message in row.get("expected_semantics", {}).get("message_types", [])]
        missing_messages = sorted(set(expected_messages) - set(entity_messages))
        if missing_messages:
            errors.append(f"{sample_id}: expected_semantics.message_types 未在实体中标注: {missing_messages}")

    return errors


def _check_value(errors: list[str], sample_id: str, field_name: str, value: Any, allowed_values: set[Any]) -> None:
    """检查单个分类标签值是否合法。"""

    if value not in allowed_values:
        errors.append(f"{sample_id}: {field_name}={value!r} 不在允许范围内")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Validate semantic extraction JSONL dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--message-knowledge", default=str(DEFAULT_MESSAGE_KNOWLEDGE))
    args = parser.parse_args()

    errors = validate_dataset(Path(args.dataset), Path(args.message_knowledge))
    if errors:
        print(f"dataset validation failed: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("dataset validation passed")


if __name__ == "__main__":
    main()
