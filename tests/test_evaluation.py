from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DATASET_PATH = EVALUATION_DIR / "eval_dataset.json"
TMP_EVALUATION_DIR = Path(__file__).resolve().parent / "_tmp_evaluation"


@pytest.fixture(autouse=True)
def clean_evaluation_tmp_dir():
    if TMP_EVALUATION_DIR.exists():
        shutil.rmtree(TMP_EVALUATION_DIR)
    TMP_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TMP_EVALUATION_DIR.exists():
        shutil.rmtree(TMP_EVALUATION_DIR)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_dataset_has_at_least_30_samples_and_four_categories() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert len(dataset) >= 30
    categories = {
        (item["gold"]["scene_type"], item["gold"]["condition_type"])
        for item in dataset
        if item["gold"].get("is_relevant", True)
    }
    assert categories == {
        ("DC", "normal"),
        ("DC", "fault"),
        ("AC", "normal"),
        ("AC", "fault"),
    }
    assert any(not item["gold"].get("is_relevant", True) for item in dataset)
    assert {"standard", "robust", "noise"} <= {item["category"] for item in dataset}


def test_eval_dataset_has_executable_gold_annotations() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    for item in dataset:
        executable_gold = item["executable_gold"]
        if not item["gold"].get("is_relevant", True):
            assert executable_gold["is_relevant"] is False
            assert executable_gold["reject_reason"]
            continue

        assert executable_gold["is_relevant"] is True
        assert executable_gold["case_id"] == item["id"]
        assert executable_gold["scene_type"] == item["gold"]["scene_type"]
        assert executable_gold["condition_type"] == item["gold"]["condition_type"]
        assert executable_gold["test_stage"] == item["test_stage"]
        assert executable_gold["preconditions"]
        assert executable_gold["steps"]
        assert executable_gold["assertions"] or item["category"] == "robust"
        assert executable_gold["cleanup_steps"]


def test_evaluate_extraction_returns_metrics_and_can_save() -> None:
    module = _load_module("evaluate_extraction", EVALUATION_DIR / "evaluate_extraction.py")
    dataset = module.load_dataset(DATASET_PATH)

    result = module.evaluate(dataset)
    output_path = TMP_EVALUATION_DIR / "extraction_results.json"
    module.save_result(result, output_path)

    assert result["total"] == len(dataset)
    assert result["relevant_total"] < result["total"]
    assert "is_relevant_accuracy" in result["metrics"]
    assert "scene_type_accuracy" in result["metrics"]
    assert output_path.exists()


def test_evaluate_generation_returns_metrics_and_can_save() -> None:
    module = _load_module("evaluate_generation", EVALUATION_DIR / "evaluate_generation.py")
    dataset = module.load_dataset(DATASET_PATH)

    result = module.evaluate(dataset)
    relevant_total = sum(1 for item in dataset if item["gold"].get("is_relevant", True))
    output_path = TMP_EVALUATION_DIR / "generation_results.json"
    module.save_result(result, output_path)

    assert result["total"] == relevant_total
    assert "stage_match_accuracy" in result["metrics"]
    assert "average_steps" in result["metrics"]
    assert output_path.exists()
