"""信息抽取评估脚本。"""

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

from ev_charge_testcase_generator.extractor import RuleBasedExtractor


def load_dataset(dataset_path: Path | str) -> list[dict[str, Any]]:
    """读取评估数据集。"""

    return json.loads(Path(dataset_path).read_text(encoding="utf-8"))


def evaluate(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """评估规则抽取效果。"""

    extractor = RuleBasedExtractor()
    counters = {
        "is_relevant": 0,
        "scene_type": 0,
        "condition_type": 0,
        "parameters": 0,
        "fault_type": 0,
        "expected_results": 0,
    }
    relevant_total = 0
    failures: list[dict[str, Any]] = []

    for item in dataset:
        gold = item["gold"]
        predicted = extractor.extract(item["text"])
        expected_relevant = bool(gold.get("is_relevant", True))
        predicted_relevant = predicted.scene_type is not None
        counters["is_relevant"] += int(predicted_relevant == expected_relevant)

        if not expected_relevant:
            if predicted_relevant != expected_relevant:
                failures.append(
                    {
                        "id": item["id"],
                        "failed_metrics": ["is_relevant"],
                        "predicted": predicted.to_dict(),
                        "gold": gold,
                    }
                )
            continue

        relevant_total += 1
        checks = {
            "scene_type": predicted.scene_type == gold["scene_type"],
            "condition_type": predicted.condition_type == gold["condition_type"],
            "parameters": predicted.parameters == gold["parameters"],
            "fault_type": predicted.fault_type == gold["fault_type"],
            "expected_results": set(gold["expected_results"]).issubset(set(predicted.expected_results)),
        }
        for metric, passed in checks.items():
            counters[metric] += int(passed)
        if not all(checks.values()):
            failures.append(
                {
                    "id": item["id"],
                    "failed_metrics": [metric for metric, passed in checks.items() if not passed],
                    "predicted": predicted.to_dict(),
                    "gold": gold,
                }
            )

    total = len(dataset)
    metrics = {
        "is_relevant_accuracy": counters["is_relevant"] / total,
        "scene_type_accuracy": counters["scene_type"] / relevant_total if relevant_total else 0.0,
        "condition_type_accuracy": counters["condition_type"] / relevant_total if relevant_total else 0.0,
        "parameters_accuracy": counters["parameters"] / relevant_total if relevant_total else 0.0,
        "fault_type_accuracy": counters["fault_type"] / relevant_total if relevant_total else 0.0,
        "expected_results_accuracy": counters["expected_results"] / relevant_total if relevant_total else 0.0,
    }
    return {"total": total, "relevant_total": relevant_total, "metrics": metrics, "failures": failures}


def print_table(result: dict[str, Any]) -> None:
    """以表格形式打印评估结果。"""

    print("+---------------------------+----------+")
    print("| Metric                    | Value    |")
    print("+---------------------------+----------+")
    print(f"| total                     | {result['total']:<8} |")
    print(f"| relevant_total            | {result['relevant_total']:<8} |")
    for metric, value in result["metrics"].items():
        print(f"| {metric:<25} | {value:<8.4f} |")
    print("+---------------------------+----------+")
    print(f"failures: {len(result['failures'])}")


def save_result(result: dict[str, Any], output_path: Path | str) -> None:
    """保存评估结果。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Evaluate rule-based extraction.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "evaluation" / "eval_dataset.json"))
    parser.add_argument("--save", action="store_true", help="Save result to evaluation/results.json")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "results.json"))
    args = parser.parse_args()

    result = evaluate(load_dataset(args.dataset))
    print_table(result)
    if args.save:
        save_result(result, args.output)


if __name__ == "__main__":
    main()
