from __future__ import annotations

import json
from pathlib import Path

from ev_charge_testcase_generator.executable_generator import ExecutableTestCaseGenerator


DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "eval_dataset.json"


def _load_executable_gold(sample_id: str) -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for item in dataset:
        if item["id"] == sample_id:
            return item["executable_gold"]
    raise AssertionError(f"sample not found: {sample_id}")


def test_generate_executable_test_case_from_standard_annotation() -> None:
    executable_gold = _load_executable_gold("STD_DC_001")

    test_case = ExecutableTestCaseGenerator().generate(executable_gold)

    assert test_case is not None
    assert test_case.case_id == "STD_DC_001"
    assert test_case.scene_type == "DC"
    assert test_case.test_stage == "低压辅助上电及充电握手阶段"
    assert test_case.steps[0].action_id == "ASET-00001-0"
    assert test_case.steps[1].action_id == "ASET-00010"
    assert test_case.assertions[0].assertion_id == "PJ-BHM-001"
    assert test_case.cleanup_steps[0].action_id == "AC-DC_STOP-00246"


def test_generate_executable_test_case_maps_timeout_wait_action() -> None:
    executable_gold = _load_executable_gold("STD_DC_009")

    test_case = ExecutableTestCaseGenerator().generate(executable_gold)

    assert test_case is not None
    assert any(step.action_id == "SLEEP-70000" for step in test_case.steps)
    assert any(assertion.assertion_id == "PJ-BEM-001" for assertion in test_case.assertions)


def test_generate_returns_none_for_noise_input() -> None:
    executable_gold = _load_executable_gold("NOISE_001")

    test_case = ExecutableTestCaseGenerator().generate(executable_gold)

    assert test_case is None
