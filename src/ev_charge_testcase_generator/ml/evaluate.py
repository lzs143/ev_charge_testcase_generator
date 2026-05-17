"""评估已训练的语义抽取模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="Evaluate a trained semantic extraction model.")
    parser.add_argument("--data", required=True, help="评估 JSONL 数据集路径")
    parser.add_argument("--model-dir", required=True, help="包含 best_model.pt、label_vocab.json 和 tokenizer 文件的目录")
    parser.add_argument("--output", required=True, help="评估指标 JSON 输出路径")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="实体阈值，默认使用 checkpoint 中保存的阈值")
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_arg_parser().parse_args()

    import torch
    from transformers import AutoTokenizer

    from ev_charge_testcase_generator.ml.entity_decoder import decode_entities
    from ev_charge_testcase_generator.ml.model import MacBertGlobalPointerForSemanticExtraction
    from ev_charge_testcase_generator.ml.predict import _decode_classification, _decode_multi_labels, _resolve_device
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
    counts = _empty_counts()

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
            _update_entity_counts(counts, row.get("entities", []), predicted_entities)
            _update_classification_counts(
                counts,
                row.get("labels", {}),
                _decode_classification(outputs["classification_logits"], label_vocab["classification_labels"]),
            )
            _update_multi_label_counts(
                counts,
                row.get("labels", {}),
                {
                    "action_intent": _decode_multi_labels(
                        outputs["multi_label_logits"].get("action_intent"),
                        label_vocab.get("multi_label_labels", {}).get("action_intent", {}),
                        threshold=threshold,
                    )
                },
            )

    metrics = _build_metrics(counts, threshold=threshold, sample_total=len(rows))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _empty_counts() -> dict[str, Any]:
    """初始化计数器。"""

    return {
        "entity_tp": 0,
        "entity_pred": 0,
        "entity_gold": 0,
        "classification_correct": {},
        "classification_total": {},
        "multi_tp": 0,
        "multi_pred": 0,
        "multi_gold": 0,
    }


def _update_entity_counts(
    counts: dict[str, Any],
    gold_entities: list[dict[str, Any]],
    predicted_entities: list[dict[str, Any]],
) -> None:
    """更新实体精确匹配计数。"""

    gold_keys = {
        (str(entity["label"]), int(entity["start"]), int(entity["end"]))
        for entity in gold_entities
    }
    pred_keys = {
        (str(entity["label"]), int(entity["start"]), int(entity["end"]))
        for entity in predicted_entities
    }
    counts["entity_tp"] += len(gold_keys & pred_keys)
    counts["entity_pred"] += len(pred_keys)
    counts["entity_gold"] += len(gold_keys)


def _update_classification_counts(
    counts: dict[str, Any],
    gold_labels: dict[str, Any],
    predicted_labels: dict[str, str],
) -> None:
    """更新单标签分类准确率计数。"""

    for field, predicted_value in predicted_labels.items():
        gold_value = _normalize_label(gold_labels.get(field))
        counts["classification_total"][field] = counts["classification_total"].get(field, 0) + 1
        if str(predicted_value) == gold_value:
            counts["classification_correct"][field] = counts["classification_correct"].get(field, 0) + 1


def _update_multi_label_counts(
    counts: dict[str, Any],
    gold_labels: dict[str, Any],
    predicted_labels: dict[str, list[str]],
) -> None:
    """更新 action_intent 多标签 micro 计数。"""

    for field, predicted_values in predicted_labels.items():
        gold_values = set(gold_labels.get(field, []))
        predicted_set = set(predicted_values)
        counts["multi_tp"] += len(gold_values & predicted_set)
        counts["multi_pred"] += len(predicted_set)
        counts["multi_gold"] += len(gold_values)


def _build_metrics(counts: dict[str, Any], threshold: float, sample_total: int) -> dict[str, Any]:
    """根据计数器生成指标。"""

    entity = _prf(counts["entity_tp"], counts["entity_pred"], counts["entity_gold"])
    multi = _prf(counts["multi_tp"], counts["multi_pred"], counts["multi_gold"])
    return {
        "sample_total": sample_total,
        "threshold": threshold,
        "entity": {
            **entity,
            "predicted": counts["entity_pred"],
            "gold": counts["entity_gold"],
            "true_positive": counts["entity_tp"],
        },
        "classification_accuracy": {
            field: counts["classification_correct"].get(field, 0) / max(total, 1)
            for field, total in counts["classification_total"].items()
        },
        "action_intent_micro_f1": multi["f1"],
        "action_intent_precision": multi["precision"],
        "action_intent_recall": multi["recall"],
    }


def _prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    """计算 micro precision/recall/f1。"""

    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _normalize_label(value: Any) -> str:
    """统一空分类标签表示。"""

    return "None" if value is None else str(value)


if __name__ == "__main__":
    main()
