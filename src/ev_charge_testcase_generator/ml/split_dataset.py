"""语义抽取数据集划分工具。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """按关键标签进行近似分层划分。"""

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratify_key(row)].append(row)

    total = len(rows)
    target_dev = round(total * dev_ratio)
    target_train = round(total * train_ratio)
    target_test = total - target_train - target_dev
    group_items = list(groups.values())
    group_plans: list[dict[str, Any]] = []
    for group_rows in group_items:
        rng.shuffle(group_rows)
        count = len(group_rows)
        dev_exact = count * dev_ratio
        test_exact = count * (1 - train_ratio - dev_ratio)
        dev_count = math.floor(dev_exact)
        test_count = math.floor(test_exact)
        if dev_count + test_count > count:
            test_count = max(0, count - dev_count)
        group_plans.append(
            {
                "rows": group_rows,
                "dev": dev_count,
                "test": test_count,
                "dev_remainder": dev_exact - dev_count,
                "test_remainder": test_exact - test_count,
            }
        )

    _distribute_remainders(group_plans, "dev", "dev_remainder", target_dev)
    _distribute_remainders(group_plans, "test", "test_remainder", target_test)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for plan in group_plans:
        group_rows = plan["rows"]
        dev_count = int(plan["dev"])
        test_count = int(plan["test"])
        splits["dev"].extend(group_rows[:dev_count])
        splits["test"].extend(group_rows[dev_count : dev_count + test_count])
        splits["train"].extend(group_rows[dev_count + test_count :])

    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)
    return splits


def _distribute_remainders(plans: list[dict[str, Any]], split_name: str, remainder_name: str, target_count: int) -> None:
    """按小数余量把样本补到目标数量。"""

    current_count = sum(int(plan[split_name]) for plan in plans)
    remaining = target_count - current_count
    if remaining <= 0:
        return
    sorted_plans = sorted(plans, key=lambda plan: float(plan[remainder_name]), reverse=True)
    index = 0
    while remaining > 0 and sorted_plans:
        plan = sorted_plans[index % len(sorted_plans)]
        used = int(plan["dev"]) + int(plan["test"])
        if used < len(plan["rows"]):
            plan[split_name] = int(plan[split_name]) + 1
            remaining -= 1
        index += 1
        if index > len(sorted_plans) * 4 and all(int(plan["dev"]) + int(plan["test"]) >= len(plan["rows"]) for plan in sorted_plans):
            break


def summarize(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """统计划分后的标签分布。"""

    return {
        split_name: {
            "total": len(rows),
            "scene_type": _counter(rows, "scene_type"),
            "condition_type": _counter(rows, "condition_type"),
            "test_stage": _counter(rows, "test_stage"),
            "fault_type": _counter(rows, "fault_type"),
            "standard_source": _counter(rows, "standard_source"),
        }
        for split_name, rows in splits.items()
    }


def _stratify_key(row: dict[str, Any]) -> str:
    """构造分层键。"""

    labels = row.get("labels", {})
    return "|".join(
        [
            str(labels.get("standard_source")),
            str(labels.get("scene_type")),
            str(labels.get("condition_type")),
            str(labels.get("fault_type")),
        ]
    )


def _counter(rows: list[dict[str, Any]], label_name: str) -> dict[str, int]:
    """统计某个句级标签。"""

    counter = Counter(str(row.get("labels", {}).get(label_name)) for row in rows)
    return dict(counter.most_common())


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Split semantic extraction JSONL dataset.")
    parser.add_argument("--input", default="data/semantic_dataset_seed.jsonl")
    parser.add_argument("--output-dir", default="data/semantic_dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    splits = split_rows(rows, args.train_ratio, args.dev_ratio, args.seed)
    output_dir = Path(args.output_dir)
    for split_name, split_rows_ in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_rows_)
    summary = summarize(splits)
    (output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: len(split_rows_) for name, split_rows_ in splits.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
