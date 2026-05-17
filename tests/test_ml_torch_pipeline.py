from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class DummyTokenizer:
    def __call__(
        self,
        text: str,
        truncation: bool,
        max_length: int,
        padding: str,
        return_offsets_mapping: bool,
        return_tensors: Any = None,
    ) -> dict[str, Any]:
        offsets = [(0, 0)]
        for index, _char in enumerate(text[: max_length - 2]):
            offsets.append((index, index + 1))
        offsets.append((0, 0))
        offsets.extend([(0, 0)] * (max_length - len(offsets)))
        input_ids = list(range(1, max_length + 1))
        return {
            "input_ids": input_ids,
            "attention_mask": [1 if offset != (0, 0) else 0 for offset in offsets],
            "token_type_ids": [0] * max_length,
            "offset_mapping": offsets,
        }


def test_tokenizer_dataset_reads_one_sample_and_maps_span(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ev_charge_testcase_generator.ml.torch_dataset import SemanticExtractionTorchDataset

    data_path, label_vocab = _write_tiny_dataset(tmp_path)

    dataset = SemanticExtractionTorchDataset(
        data_path,
        label_vocab=label_vocab,
        max_length=16,
        tokenizer=DummyTokenizer(),
    )
    sample = dataset[0]

    assert sample["text"] == "发送BRM"
    assert sample["entity_labels"].shape == (2, 16, 16)
    assert sample["token_spans"]


def test_token_span_mapping_is_in_bounds(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ev_charge_testcase_generator.ml.torch_dataset import SemanticExtractionTorchDataset

    data_path, label_vocab = _write_tiny_dataset(tmp_path)
    sample = SemanticExtractionTorchDataset(data_path, label_vocab, max_length=16, tokenizer=DummyTokenizer())[0]

    for token_span in sample["token_spans"]:
        assert 0 <= token_span.start <= token_span.end < 16


def test_model_forward_runs_on_small_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ev_charge_testcase_generator.ml import model as model_module

    class DummyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> Any:
            batch_size, seq_len = input_ids.shape
            return SimpleNamespace(last_hidden_state=torch.ones(batch_size, seq_len, 8))

    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", lambda _name, **_kwargs: DummyEncoder())
    semantic_model = model_module.MacBertGlobalPointerForSemanticExtraction.from_label_vocab("dummy", _tiny_label_vocab())
    outputs = semantic_model(
        input_ids=torch.ones(2, 8, dtype=torch.long),
        attention_mask=torch.ones(2, 8, dtype=torch.long),
        token_type_ids=torch.zeros(2, 8, dtype=torch.long),
        entity_labels=torch.zeros(2, 2, 8, 8),
        classification_labels={
            field: torch.zeros(2, dtype=torch.long)
            for field in _tiny_label_vocab()["classification_labels"]
        },
        multi_label_labels={"action_intent": torch.zeros(2, 2)},
    )

    assert outputs["entity_logits"].shape == (2, 2, 8, 8)
    assert "loss" in outputs


def test_train_help_displays() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [sys.executable, "-m", "ev_charge_testcase_generator.ml.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "--model-name" in result.stdout


def _write_tiny_dataset(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    row = {
        "id": "tiny-1",
        "text": "发送BRM",
        "entities": [
            {"start": 0, "end": 2, "label": "ACTION", "text": "发送"},
            {"start": 2, "end": 5, "label": "MESSAGE", "text": "BRM"},
        ],
        "labels": {
            "standard_source": "GB/T 34658-2025",
            "scene_type": "DC",
            "system_class": "unknown",
            "test_layer": "application_layer",
            "condition_type": "normal",
            "test_type": "positive",
            "test_stage": "握手阶段",
            "fault_type": None,
            "action_intent": ["发送报文"],
        },
    }
    data_path = tmp_path / "tiny.jsonl"
    data_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return data_path, _tiny_label_vocab()


def _tiny_label_vocab() -> dict[str, Any]:
    return {
        "entity_labels": {"ACTION": 0, "MESSAGE": 1},
        "classification_labels": {
            "standard_source": {"GB/T 34658-2025": 0},
            "scene_type": {"DC": 0},
            "system_class": {"unknown": 0},
            "test_layer": {"application_layer": 0},
            "condition_type": {"normal": 0},
            "test_type": {"positive": 0},
            "test_stage": {"握手阶段": 0},
            "fault_type": {"None": 0},
        },
        "multi_label_labels": {"action_intent": {"发送报文": 0, "检查响应": 1}},
    }
