"""实体预测解码与后处理规则。"""

from __future__ import annotations

import re
from typing import Any


ACTION_WORDS: tuple[str, ...] = (
    "发送",
    "检查",
    "设置",
    "置为",
    "注入",
    "断开",
    "闭合",
    "启动",
    "停止",
    "进入",
    "回复",
    "记录",
    "提示",
    "检测",
    "确认",
    "插入",
)
OBJECT_WORDS: tuple[str, ...] = ("BMS", "车辆", "整车", "充电机", "充电桩", "充电枪", "系统", "报文")
MESSAGE_PATTERN = re.compile(r"[A-Z]{2,4}")


def decode_entities(
    text: str,
    offsets: list[tuple[int, int]],
    logits: Any,
    entity_label_map: dict[str, int],
    threshold: float,
    message_label_map: dict[str, int] | None = None,
    post_process: bool = True,
) -> list[dict[str, Any]]:
    """把 token span logits 解码为字符级实体，并按业务词形做轻量后处理。"""

    id_to_label = {index: label for label, index in entity_label_map.items()}
    entities: list[dict[str, Any]] = []
    positive = (logits > threshold).nonzero(as_tuple=False).tolist()
    for label_id, token_start, token_end in positive:
        char_start, _ = offsets[token_start]
        _, char_end = offsets[token_end]
        if char_start == char_end or char_end <= char_start:
            continue
        entities.append(
            {
                "label": id_to_label[int(label_id)],
                "start": char_start,
                "end": char_end,
                "text": text[char_start:char_end],
                "score": float(logits[label_id, token_start, token_end]),
            }
        )
    if post_process:
        entities = post_process_entities(text, entities, message_label_map=message_label_map)
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    return entities


def post_process_entities(
    text: str,
    entities: list[dict[str, Any]],
    message_label_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """根据充电测试领域词形约束收紧过长实体 span。"""

    normalized: list[dict[str, Any]] = []
    message_names = set(message_label_map or {})
    for entity in entities:
        label = str(entity["label"])
        if label == "MESSAGE":
            normalized.extend(_extract_message_entities(text, entity, message_names))
        elif label == "ACTION":
            normalized.extend(_extract_keyword_entities(text, entity, ACTION_WORDS))
        elif label == "OBJECT":
            normalized.extend(_extract_keyword_entities(text, entity, OBJECT_WORDS))
        else:
            normalized.append(entity)
    return _suppress_overlaps(normalized)


def _extract_message_entities(text: str, entity: dict[str, Any], message_names: set[str]) -> list[dict[str, Any]]:
    """从过长 MESSAGE span 中抽取真实报文缩写。"""

    results: list[dict[str, Any]] = []
    for match in MESSAGE_PATTERN.finditer(str(entity["text"])):
        message = match.group()
        if message_names and message not in message_names:
            continue
        start = int(entity["start"]) + match.start()
        end = int(entity["start"]) + match.end()
        results.append({**entity, "start": start, "end": end, "text": text[start:end]})
    return results or [entity]


def _extract_keyword_entities(text: str, entity: dict[str, Any], keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    """从过长实体中截取动作词或对象词。"""

    results: list[dict[str, Any]] = []
    entity_text = str(entity["text"])
    for keyword in keywords:
        search_start = 0
        while True:
            offset = entity_text.find(keyword, search_start)
            if offset < 0:
                break
            start = int(entity["start"]) + offset
            end = start + len(keyword)
            results.append({**entity, "start": start, "end": end, "text": text[start:end]})
            search_start = offset + len(keyword)
    return results or [entity]


def _suppress_overlaps(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉同类型重叠实体，优先保留分数高且更短的候选。"""

    selected: list[dict[str, Any]] = []
    for entity in sorted(
        entities,
        key=lambda item: (
            str(item["label"]),
            -float(item.get("score", 0.0)),
            int(item["end"]) - int(item["start"]),
        ),
    ):
        if any(_same_label_overlap(entity, kept) for kept in selected):
            continue
        selected.append(entity)
    return selected


def _same_label_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两个同类型实体是否有字符重叠。"""

    if left["label"] != right["label"]:
        return False
    return min(int(left["end"]), int(right["end"])) > max(int(left["start"]), int(right["start"]))
