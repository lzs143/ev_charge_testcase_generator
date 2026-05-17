from __future__ import annotations

import json
from pathlib import Path

from ev_charge_testcase_generator.ml.dataset import encode_sample
from ev_charge_testcase_generator.ml.label_vocab import build_label_vocab


def test_label_vocab_contains_entity_and_classification_labels() -> None:
    rows = _load_rows("data/semantic_dataset/train.jsonl")

    vocab = build_label_vocab(rows)

    assert "MESSAGE" in vocab["entity_labels"]
    assert "ACTION" in vocab["entity_labels"]
    assert "scene_type" in vocab["classification_labels"]
    assert "DC" in vocab["classification_labels"]["scene_type"]
    assert "发送报文" in vocab["multi_label_labels"]["action_intent"]


def test_encode_sample_preserves_entity_span_and_labels() -> None:
    rows = _load_rows("data/semantic_dataset/train.jsonl")
    vocab = build_label_vocab(rows)
    row = next(item for item in rows if item["entities"])

    encoded = encode_sample(row, vocab)

    assert encoded["id"] == row["id"]
    assert encoded["tokens"] == list(row["text"])
    assert encoded["entity_spans"]
    first_span = encoded["entity_spans"][0]
    assert row["text"][first_span["start"] : first_span["end"] + 1] == first_span["text"]
    assert "scene_type" in encoded["classification_labels"]
    assert len(encoded["multi_label_labels"]["action_intent"]) == len(vocab["multi_label_labels"]["action_intent"])


def _load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

