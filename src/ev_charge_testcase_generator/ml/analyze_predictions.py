"""分析 dev 集实体抽取错误，辅助改进语义抽取模型。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="Analyze entity extraction errors on a semantic JSONL dataset.")
    parser.add_argument("--data", required=True, help="待分析的 JSONL 数据集，一般使用 dev.jsonl")
    parser.add_argument("--model-dir", required=True, help="包含 best_model.pt、label_vocab.json 和 tokenizer 文件的目录")
    parser.add_argument("--output", required=True, help="错误分析 JSON 输出路径")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="实体阈值，默认使用 checkpoint 中保存的阈值")
    parser.add_argument("--max-length", type=int, default=None, help="覆盖 checkpoint 中的 max_length")
    parser.add_argument("--max-examples", type=int, default=30, help="保存的错误样例数量上限")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_arg_parser().parse_args()

    import torch
    from transformers import AutoTokenizer

    from ev_charge_testcase_generator.ml.entity_decoder import decode_entities
    from ev_charge_testcase_generator.ml.model import MacBertGlobalPointerForSemanticExtraction
    from ev_charge_testcase_generator.ml.predict import _resolve_device
    from ev_charge_testcase_generator.ml.torch_dataset import load_jsonl

    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")
    label_vocab = json.loads((model_dir / "label_vocab.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_name = str(checkpoint.get("model_name", "hfl/chinese-macbert-base"))
    max_length = int(args.max_length or checkpoint.get("max_length", 256))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("entity_threshold", 0.0))

    device = _resolve_device(args.device, torch)
    model = MacBertGlobalPointerForSemanticExtraction.from_label_vocab(
        model_name,
        label_vocab,
        local_files_only=True,
        init_from_config_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows = load_jsonl(args.data)
    sample_reports: list[dict[str, Any]] = []
    summary = _empty_summary(threshold=threshold)

    with torch.no_grad():
        for row in rows:
            text = str(row["text"])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = [(int(start), int(end)) for start, end in encoded.pop("offset_mapping")[0].tolist()]
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            predicted_entities = decode_entities(
                text,
                offsets,
                outputs["entity_logits"][0].detach().cpu(),
                label_vocab["entity_labels"],
                threshold=threshold,
                message_label_map=label_vocab.get("message_labels", {}),
            )
            gold_entities = _normalize_gold_entities(row.get("entities", []))
            report = _analyze_sample(
                sample_id=str(row.get("id", "")),
                text=text,
                gold_entities=gold_entities,
                predicted_entities=predicted_entities,
            )
            _merge_summary(summary, report)
            if _has_errors(report) and len(sample_reports) < args.max_examples:
                sample_reports.append(report)

    summary["precision"] = _safe_div(summary["exact_match"], summary["predicted_total"])
    summary["recall"] = _safe_div(summary["exact_match"], summary["gold_total"])
    summary["f1"] = _f1(summary["precision"], summary["recall"])
    output = {
        "summary": summary,
        "error_examples": sample_reports,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _empty_summary(threshold: float) -> dict[str, Any]:
    """初始化聚合统计结构。"""

    return {
        "threshold": threshold,
        "sample_total": 0,
        "gold_total": 0,
        "predicted_total": 0,
        "exact_match": 0,
        "false_positive_total": 0,
        "false_negative_total": 0,
        "boundary_error_total": 0,
        "type_error_total": 0,
        "false_positive_by_label": {},
        "false_negative_by_label": {},
        "boundary_error_by_label": {},
        "type_confusion": {},
    }


def _normalize_gold_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把原始标注实体规范为统一结构。"""

    return [
        {
            "label": str(entity["label"]),
            "start": int(entity["start"]),
            "end": int(entity["end"]),
            "text": str(entity["text"]),
        }
        for entity in entities
    ]


def _analyze_sample(
    sample_id: str,
    text: str,
    gold_entities: list[dict[str, Any]],
    predicted_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """分析单条样本的实体预测错误。"""

    gold_keys = {_entity_key(entity) for entity in gold_entities}
    pred_keys = {_entity_key(entity) for entity in predicted_entities}
    exact_keys = gold_keys & pred_keys
    false_positive = [entity for entity in predicted_entities if _entity_key(entity) not in gold_keys]
    false_negative = [entity for entity in gold_entities if _entity_key(entity) not in pred_keys]

    boundary_errors: list[dict[str, Any]] = []
    type_errors: list[dict[str, Any]] = []
    for pred in false_positive:
        same_span_gold = _find_same_span(pred, false_negative)
        if same_span_gold is not None:
            type_errors.append({"predicted": pred, "gold": same_span_gold})
            continue
        overlapping_gold = _find_best_overlap(pred, false_negative, same_label=True)
        if overlapping_gold is not None:
            boundary_errors.append({"predicted": pred, "gold": overlapping_gold})

    return {
        "id": sample_id,
        "text": text,
        "gold": gold_entities,
        "predicted": predicted_entities,
        "exact_match": len(exact_keys),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "boundary_errors": boundary_errors,
        "type_errors": type_errors,
    }


def _merge_summary(summary: dict[str, Any], report: dict[str, Any]) -> None:
    """把单条样本报告合并到全局统计。"""

    false_positive = report["false_positive"]
    false_negative = report["false_negative"]
    boundary_errors = report["boundary_errors"]
    type_errors = report["type_errors"]
    summary["sample_total"] += 1
    summary["gold_total"] += len(report["gold"])
    summary["predicted_total"] += len(report["predicted"])
    summary["exact_match"] += int(report["exact_match"])
    summary["false_positive_total"] += len(false_positive)
    summary["false_negative_total"] += len(false_negative)
    summary["boundary_error_total"] += len(boundary_errors)
    summary["type_error_total"] += len(type_errors)

    _update_counter(summary, "false_positive_by_label", Counter(entity["label"] for entity in false_positive))
    _update_counter(summary, "false_negative_by_label", Counter(entity["label"] for entity in false_negative))
    _update_counter(summary, "boundary_error_by_label", Counter(item["gold"]["label"] for item in boundary_errors))
    _update_counter(
        summary,
        "type_confusion",
        Counter(f'{item["gold"]["label"]}->{item["predicted"]["label"]}' for item in type_errors),
    )


def _update_counter(summary: dict[str, Any], field: str, counter: Counter[str]) -> None:
    """累加 Counter 并按数量降序保存为普通字典。"""

    current = Counter(summary[field])
    current.update(counter)
    summary[field] = dict(current.most_common())


def _entity_key(entity: dict[str, Any]) -> tuple[str, int, int]:
    """实体精确匹配键：类型、起点、终点都必须一致。"""

    return str(entity["label"]), int(entity["start"]), int(entity["end"])


def _find_same_span(predicted: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """查找边界相同但类型可能不同的标注实体。"""

    for candidate in candidates:
        if int(candidate["start"]) == int(predicted["start"]) and int(candidate["end"]) == int(predicted["end"]):
            return candidate
    return None


def _find_best_overlap(
    predicted: dict[str, Any],
    candidates: list[dict[str, Any]],
    same_label: bool,
) -> dict[str, Any] | None:
    """查找和预测实体重叠最多的标注实体。"""

    best_candidate: dict[str, Any] | None = None
    best_overlap = 0
    for candidate in candidates:
        if same_label and candidate["label"] != predicted["label"]:
            continue
        overlap = min(int(predicted["end"]), int(candidate["end"])) - max(int(predicted["start"]), int(candidate["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = candidate
    return best_candidate


def _has_errors(report: dict[str, Any]) -> bool:
    """判断样本是否存在任意实体错误。"""

    return bool(report["false_positive"] or report["false_negative"])


def _safe_div(numerator: int, denominator: int) -> float:
    """安全除法。"""

    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """计算 F1。"""

    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


if __name__ == "__main__":
    main()
