"""语义抽取模型数据编码工具。

当前实现为纯 Python 字符级编码，目的是验证标签、span 和分类目标。
后续接入 MacBERT 时，可以用 tokenizer 的 offset_mapping 替换 char_tokens。
"""

from __future__ import annotations

import argparse
import json
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_label_vocab(path: Path) -> dict[str, Any]:
    """读取标签词表。"""

    return json.loads(path.read_text(encoding="utf-8"))


def encode_sample(row: dict[str, Any], label_vocab: dict[str, Any], max_length: int = 256) -> dict[str, Any]:
    """将单条样本编码为模型训练所需的目标结构。"""

    text = str(row["text"])
    tokens = list(text[:max_length])
    encoded: dict[str, Any] = {
        "id": row["id"],
        "text": text,
        "tokens": tokens,
        "attention_mask": [1] * len(tokens),
        "entity_spans": [],
        "classification_labels": {},
        "multi_label_labels": {},
        "message_labels": _multi_hot(
            row.get("expected_semantics", {}).get("message_types", []),
            label_vocab.get("message_labels", {}),
        ),
    }

    entity_label_map = label_vocab["entity_labels"]
    for entity in row.get("entities", []):
        start = int(entity["start"])
        end = int(entity["end"])
        if end > max_length:
            continue
        encoded["entity_spans"].append(
            {
                "label": entity["label"],
                "label_id": entity_label_map[entity["label"]],
                "start": start,
                "end": end - 1,
                "text": entity["text"],
            }
        )

    labels = row.get("labels", {})
    for field_name in CLASSIFICATION_FIELDS:
        label_value = _normalize_label(labels.get(field_name))
        encoded["classification_labels"][field_name] = label_vocab["classification_labels"][field_name][label_value]

    for field_name, label_map in label_vocab.get("multi_label_labels", {}).items():
        encoded["multi_label_labels"][field_name] = _multi_hot(labels.get(field_name, []), label_map)

    return encoded


def encode_dataset(rows: list[dict[str, Any]], label_vocab: dict[str, Any], max_length: int = 256) -> list[dict[str, Any]]:
    """编码整个数据集。"""

    return [encode_sample(row, label_vocab, max_length=max_length) for row in rows]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def _multi_hot(values: list[str], label_map: dict[str, int]) -> list[int]:
    """构造多标签 multi-hot 向量。"""

    vector = [0] * len(label_map)
    for value in values:
        if value in label_map:
            vector[label_map[value]] = 1
    return vector


def _normalize_label(value: Any) -> str:
    """统一分类标签中的空值表示。"""

    return "None" if value is None else str(value)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Encode semantic extraction dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--label-vocab", default="data/semantic_dataset/label_vocab.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    label_vocab = load_label_vocab(Path(args.label_vocab))
    encoded_rows = encode_dataset(rows, label_vocab, max_length=args.max_length)
    write_jsonl(Path(args.output), encoded_rows)
    print(json.dumps({"total": len(encoded_rows), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
