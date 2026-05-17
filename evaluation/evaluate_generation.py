"""自然语言到可执行测试用例生成评估脚本。"""

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

from ev_charge_testcase_generator.pipeline import TestCaseGenerationPipeline


def load_dataset(dataset_path: Path | str) -> list[dict[str, Any]]:
    """读取评估数据集。"""

    return json.loads(Path(dataset_path).read_text(encoding="utf-8"))


def evaluate(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """评估自然语言到可执行测试用例的端到端生成效果。"""

    pipeline = TestCaseGenerationPipeline()
    relevant_items = [item for item in dataset if item["gold"].get("is_relevant", True)]
    total = len(relevant_items)
    generation_successes = 0
    check_passes = 0
    stage_matches = 0
    total_steps = 0
    total_assertions = 0
    failures: list[dict[str, Any]] = []

    for item in relevant_items:
        try:
            result = pipeline.run(item["text"])
        except Exception as exc:  # noqa: BLE001 - 评估脚本需要记录失败案例
            failures.append({"id": item["id"], "reason": str(exc)})
            continue

        generation_successes += 1
        total_steps += len(result.test_case.steps)
        total_assertions += len(result.test_case.assertions)
        if result.check_result.passed:
            check_passes += 1
        else:
            failures.append(
                {
                    "id": item["id"],
                    "reason": "check_failed",
                    "errors": result.check_result.errors,
                    "warnings": result.check_result.warnings,
                }
            )

        expected_stage = item.get("test_stage")
        if result.test_case.test_stage == expected_stage:
            stage_matches += 1
        else:
            failures.append(
                {
                    "id": item["id"],
                    "reason": "stage_mismatch",
                    "expected": expected_stage,
                    "predicted": result.test_case.test_stage,
                }
            )

    metrics = {
        "generation_success_rate": generation_successes / total if total else 0.0,
        "check_pass_rate": check_passes / total if total else 0.0,
        "stage_match_accuracy": stage_matches / total if total else 0.0,
        "average_steps": total_steps / generation_successes if generation_successes else 0.0,
        "average_assertions": total_assertions / generation_successes if generation_successes else 0.0,
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

    parser = argparse.ArgumentParser(description="Evaluate natural language to executable testcase generation.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "evaluation" / "eval_dataset.json"))
    parser.add_argument("--save", action="store_true", help="Save result to evaluation/generation_results.json")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "generation_results.json"))
    args = parser.parse_args()

    result = evaluate(load_dataset(args.dataset))
    print_table(result)
    if args.save:
        save_result(result, args.output)


if __name__ == "__main__":
    main()
