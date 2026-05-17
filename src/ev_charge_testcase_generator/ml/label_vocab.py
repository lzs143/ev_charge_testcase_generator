"""语义抽取模型标签词表生成。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLASSIFICATION_FIELDS = (
    "standard_source",
    "scene_type",
    "system_class",
    "test_layer",
    "condition_type",
    "test_type",
    "test_stage",
    "fault_type",
)
MULTI_LABEL_FIELDS = ("action_intent",)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_label_vocab(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """从语义数据集中构建标签词表。"""

    entity_labels = sorted({entity["label"] for row in rows for entity in row.get("entities", [])})
    vocab: dict[str, Any] = {
        "entity_labels": _to_index(entity_labels),
        "classification_labels": {},
        "multi_label_labels": {},
        "message_labels": _to_index(sorted({message for row in rows for message in row.get("expected_semantics", {}).get("message_types", [])})),
        "statistics": {
            "total": len(rows),
            "entity_label_count": len(entity_labels),
            "classification": {},
            "multi_label": {},
        },
    }

    for field_name in CLASSIFICATION_FIELDS:
        values = sorted({_normalize_label(row.get("labels", {}).get(field_name)) for row in rows})
        vocab["classification_labels"][field_name] = _to_index(values)
        vocab["statistics"]["classification"][field_name] = dict(
            Counter(_normalize_label(row.get("labels", {}).get(field_name)) for row in rows).most_common()
        )

    for field_name in MULTI_LABEL_FIELDS:
        values = sorted({value for row in rows for value in row.get("labels", {}).get(field_name, [])})
        vocab["multi_label_labels"][field_name] = _to_index(values)
        vocab["statistics"]["multi_label"][field_name] = dict(
            Counter(value for row in rows for value in row.get("labels", {}).get(field_name, [])).most_common()
        )

    return vocab


def save_label_vocab(vocab: dict[str, Any], output_path: Path) -> None:
    """保存标签词表。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_index(values: list[str]) -> dict[str, int]:
    """把标签列表转换为稳定 id 映射。"""

    return {value: index for index, value in enumerate(values)}


def _normalize_label(value: Any) -> str:
    """统一分类标签中的空值表示。"""

    return "None" if value is None else str(value)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Build semantic extraction label vocabulary.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/semantic_dataset/train.jsonl",
            "data/semantic_dataset/dev.jsonl",
            "data/semantic_dataset/test.jsonl",
        ],
    )
    parser.add_argument("--output", default="data/semantic_dataset/label_vocab.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        rows.extend(load_jsonl(Path(input_path)))
    vocab = build_label_vocab(rows)
    save_label_vocab(vocab, Path(args.output))
    print(
        json.dumps(
            {
                "total": vocab["statistics"]["total"],
                "entity_labels": len(vocab["entity_labels"]),
                "messages": len(vocab["message_labels"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
