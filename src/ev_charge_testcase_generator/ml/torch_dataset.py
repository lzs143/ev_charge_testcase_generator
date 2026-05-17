"""基于 transformers tokenizer 的语义抽取训练数据集。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ev_charge_testcase_generator.ml.dataset import CLASSIFICATION_FIELDS


@dataclass(frozen=True)
class TokenSpan:
    """字符级实体映射后的 token 闭区间。"""

    label_id: int
    start: int
    end: int


class SemanticExtractionTorchDataset(Dataset[dict[str, Any]]):
    """直接读取原始 JSONL，并构造 GlobalPointer 与多任务分类标签。"""

    def __init__(
        self,
        jsonl_path: str | Path,
        label_vocab: dict[str, Any],
        model_name: str = "hfl/chinese-macbert-base",
        max_length: int = 256,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.path = Path(jsonl_path)
        self.rows = load_jsonl(self.path)
        self.label_vocab = label_vocab
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        text = str(row["text"])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors=None,
        )
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        entity_labels = torch.zeros(
            (len(self.label_vocab["entity_labels"]), self.max_length, self.max_length),
            dtype=torch.float,
        )
        token_spans: list[TokenSpan] = []
        for entity in row.get("entities", []):
            token_span = map_char_span_to_token_span(
                offsets=offsets,
                char_start=int(entity["start"]),
                char_end=int(entity["end"]),
                label_id=int(self.label_vocab["entity_labels"][entity["label"]]),
            )
            if token_span is None:
                continue
            entity_labels[token_span.label_id, token_span.start, token_span.end] = 1.0
            token_spans.append(token_span)

        labels = row.get("labels", {})
        classification_labels = {
            field: torch.tensor(
                self.label_vocab["classification_labels"][field][_normalize_label(labels.get(field))],
                dtype=torch.long,
            )
            for field in CLASSIFICATION_FIELDS
            if field in self.label_vocab.get("classification_labels", {})
        }
        multi_label_labels = {
            field: torch.tensor(_multi_hot(labels.get(field, []), label_map), dtype=torch.float)
            for field, label_map in self.label_vocab.get("multi_label_labels", {}).items()
        }

        return {
            "id": row.get("id", str(index)),
            "text": text,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoded.get("token_type_ids", [0] * self.max_length), dtype=torch.long),
            "offset_mapping": torch.tensor(offsets, dtype=torch.long),
            "entity_labels": entity_labels,
            "classification_labels": classification_labels,
            "multi_label_labels": multi_label_labels,
            "token_spans": token_spans,
        }


def semantic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """合并 batch，保留文本和 span 调试信息，张量字段按第一维堆叠。"""

    classification_fields = batch[0]["classification_labels"].keys()
    multi_label_fields = batch[0]["multi_label_labels"].keys()
    return {
        "id": [item["id"] for item in batch],
        "text": [item["text"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "token_type_ids": torch.stack([item["token_type_ids"] for item in batch]),
        "offset_mapping": torch.stack([item["offset_mapping"] for item in batch]),
        "entity_labels": torch.stack([item["entity_labels"] for item in batch]),
        "classification_labels": {
            field: torch.stack([item["classification_labels"][field] for item in batch])
            for field in classification_fields
        },
        "multi_label_labels": {
            field: torch.stack([item["multi_label_labels"][field] for item in batch])
            for field in multi_label_fields
        },
        "token_spans": [item["token_spans"] for item in batch],
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取原始语义数据 JSONL。"""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_label_vocab(path: str | Path) -> dict[str, Any]:
    """读取标签词表。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def map_char_span_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
    label_id: int,
) -> TokenSpan | None:
    """把字符级 [start, end) 实体区间映射到 tokenizer token 闭区间。"""

    token_start: int | None = None
    token_end: int | None = None
    for token_index, (offset_start, offset_end) in enumerate(offsets):
        if offset_start == offset_end:
            continue
        if token_start is None and offset_start <= char_start < offset_end:
            token_start = token_index
        if offset_start < char_end <= offset_end:
            token_end = token_index
            break
        if offset_start < char_end and offset_end <= char_end:
            token_end = token_index
    if token_start is None or token_end is None or token_start > token_end:
        return None
    return TokenSpan(label_id=label_id, start=token_start, end=token_end)


def _multi_hot(values: list[str], label_map: dict[str, int]) -> list[float]:
    """构造多标签分类的 multi-hot 向量。"""

    vector = [0.0] * len(label_map)
    for value in values:
        if value in label_map:
            vector[label_map[value]] = 1.0
    return vector


def _normalize_label(value: Any) -> str:
    """统一分类标签中的空值表示。"""

    return "None" if value is None else str(value)
