"""可执行测试用例生成评估脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ev_charge_testcase_generator.executable_generator import ExecutableTestCaseGenerator


def load_dataset(dataset_path: Path | str) -> list[dict[str, Any]]:
    """读取评估数据集。"""

    return json.loads(Path(dataset_path).read_text(encoding="utf-8"))


def evaluate(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """评估可执行测试用例生成效果。"""

    generator = ExecutableTestCaseGenerator()
    relevant_items = [item for item in dataset if item["executable_gold"].get("is_relevant", True)]
    total = len(relevant_items)
    generated = 0
    total_steps = 0
    mapped_steps = 0
    total_assertions = 0
    mapped_assertions = 0
    failures: list[dict[str, Any]] = []

    for item in relevant_items:
        test_case = generator.generate(item["executable_gold"])
        if test_case is None:
            failures.append({"id": item["id"], "reason": "not_generated"})
            continue

        generated += 1
        total_steps += len(test_case.steps) + len(test_case.cleanup_steps)
        mapped_steps += sum(1 for step in test_case.steps if step.action_id != "UNMAPPED")
        mapped_steps += sum(1 for step in test_case.cleanup_steps if step.action_id != "UNMAPPED")
        total_assertions += len(test_case.assertions)
        mapped_assertions += sum(
            1
            for assertion in test_case.assertions
            if assertion.assertion_id != "PJ-UNMAPPED"
        )

        if any(step.action_id == "UNMAPPED" for step in [*test_case.steps, *test_case.cleanup_steps]):
            failures.append({"id": item["id"], "reason": "unmapped_action"})
        if any(assertion.assertion_id == "PJ-UNMAPPED" for assertion in test_case.assertions):
            failures.append({"id": item["id"], "reason": "unmapped_assertion"})

    metrics = {
        "generation_success_rate": generated / total if total else 0.0,
        "action_mapping_rate": mapped_steps / total_steps if total_steps else 0.0,
        "assertion_mapping_rate": mapped_assertions / total_assertions if total_assertions else 0.0,
    }
    return {"total": total, "metrics": metrics, "failures": failures}


def print_table(result: dict[str, Any]) -> None:
    """以表格形式打印评估结果。"""

    print("+-------------------------+----------+")
    print("| Metric                  | Value    |")
    print("+-------------------------+----------+")
    print(f"| total                   | {result['total']:<8} |")
    for metric, value in result["metrics"].items():
        print(f"| {metric:<23} | {value:<8.4f} |")
    print("+-------------------------+----------+")
    print(f"failures: {len(result['failures'])}")


def save_result(result: dict[str, Any], output_path: Path | str) -> None:
    """保存评估结果。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Evaluate executable testcase generation.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "evaluation" / "eval_dataset.json"))
    parser.add_argument("--save", action="store_true", help="Save result to evaluation/executable_results.json")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "executable_results.json"))
    args = parser.parse_args()

    result = evaluate(load_dataset(args.dataset))
    print_table(result)
    if args.save:
        save_result(result, args.output)


if __name__ == "__main__":
    main()
