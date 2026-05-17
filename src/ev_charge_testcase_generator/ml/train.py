"""MacBERT-GlobalPointer 多任务语义抽取训练入口。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数，便于测试 --help。"""

    parser = argparse.ArgumentParser(description="Train MacBERT-GlobalPointer semantic extraction model.")
    parser.add_argument("--train", required=True, help="训练集 JSONL 路径")
    parser.add_argument("--dev", required=True, help="验证集 JSONL 路径")
    parser.add_argument("--label-vocab", required=True, help="标签词表 JSON 路径")
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base", help="HuggingFace 预训练模型名称")
    parser.add_argument("--output-dir", required=True, help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda 或 torch 支持的设备名")
    parser.add_argument("--entity-loss-weight", type=float, default=5.0, help="实体抽取 loss 在总 loss 中的权重")
    parser.add_argument("--entity-positive-weight", type=float, default=20.0, help="实体 span 正样本 BCE 权重")
    parser.add_argument(
        "--entity-thresholds",
        default="-3,-2,-1,0",
        help="dev 评估时扫描的实体 logits 阈值，逗号分隔",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    parser = build_arg_parser()
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from ev_charge_testcase_generator.ml.entity_decoder import decode_entities
    from ev_charge_testcase_generator.ml.model import MacBertGlobalPointerForSemanticExtraction
    from ev_charge_testcase_generator.ml.torch_dataset import (
        SemanticExtractionTorchDataset,
        load_label_vocab,
        semantic_collate_fn,
    )

    label_vocab = load_label_vocab(args.label_vocab)
    train_dataset = SemanticExtractionTorchDataset(
        args.train,
        label_vocab=label_vocab,
        model_name=args.model_name,
        max_length=args.max_length,
    )
    dev_dataset = SemanticExtractionTorchDataset(
        args.dev,
        label_vocab=label_vocab,
        model_name=args.model_name,
        max_length=args.max_length,
        tokenizer=train_dataset.tokenizer,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=semantic_collate_fn,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=semantic_collate_fn)

    device = _resolve_device(args.device, torch)
    model = MacBertGlobalPointerForSemanticExtraction.from_label_vocab(args.model_name, label_vocab).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_score = -1.0
    history: list[dict[str, Any]] = []
    entity_thresholds = _parse_thresholds(args.entity_thresholds)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            outputs = model(
                **_move_batch_to_device(batch, device),
                entity_loss_weight=args.entity_loss_weight,
                entity_positive_weight=args.entity_positive_weight,
            )
            loss = outputs["loss"]
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        metrics = evaluate(
            model,
            dev_loader,
            device,
            torch,
            label_vocab=label_vocab,
            entity_thresholds=entity_thresholds,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(len(train_loader), 1)
        metrics["selection_score"] = _selection_score(metrics)
        history.append(metrics)

        checkpoint = {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "entity_threshold": metrics["entity"]["threshold"],
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, output_dir / "last_model.pt")

        if metrics["entity"]["f1"] > best_f1 or (
            metrics["entity"]["f1"] == best_f1 and metrics["selection_score"] > best_score
        ):
            best_f1 = metrics["entity"]["f1"]
            best_score = metrics["selection_score"]
            torch.save(checkpoint, output_dir / "best_model.pt")

        (output_dir / "training_metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False))

    shutil.copyfile(args.label_vocab, output_dir / "label_vocab.json")
    train_dataset.tokenizer.save_pretrained(output_dir)
    print(json.dumps({"best_entity_f1": best_f1, "output_dir": str(output_dir)}, ensure_ascii=False))


def evaluate(
    model: Any,
    data_loader: Any,
    device: Any,
    torch_module: Any,
    label_vocab: dict[str, Any],
    entity_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """在验证集上计算实体、单分类、多标签指标。"""

    model.eval()
    thresholds = entity_thresholds or [0.0]
    entity_counts = {threshold: {"tp": 0, "pred": 0, "gold": 0} for threshold in thresholds}
    classification_correct: dict[str, int] = {}
    classification_total: dict[str, int] = {}
    multi_tp = multi_pred = multi_gold = 0

    with torch_module.no_grad():
        for batch in data_loader:
            outputs = model(**_move_batch_to_device(batch, device))
            entity_logits = outputs["entity_logits"]
            classification_logits = outputs["classification_logits"]
            multi_label_logits = outputs["multi_label_logits"]

            gold_entities_by_sample = _gold_entities_from_batch(batch, label_vocab)
            for threshold in thresholds:
                for sample_index, text in enumerate(batch["text"]):
                    offsets = [
                        (int(start), int(end))
                        for start, end in batch["offset_mapping"][sample_index].detach().cpu().tolist()
                    ]
                    predicted_entities = decode_entities(
                        text,
                        offsets,
                        entity_logits[sample_index].detach().cpu(),
                        label_vocab["entity_labels"],
                        threshold=threshold,
                        message_label_map=label_vocab.get("message_labels", {}),
                    )
                    pred_keys = {_entity_key(entity) for entity in predicted_entities}
                    gold_keys = gold_entities_by_sample[sample_index]
                    entity_counts[threshold]["tp"] += len(pred_keys & gold_keys)
                    entity_counts[threshold]["pred"] += len(pred_keys)
                    entity_counts[threshold]["gold"] += len(gold_keys)

            for field, logits in classification_logits.items():
                gold = batch["classification_labels"][field].to(device)
                pred = logits.argmax(dim=-1)
                classification_correct[field] = classification_correct.get(field, 0) + int((pred == gold).sum().detach().cpu())
                classification_total[field] = classification_total.get(field, 0) + int(gold.numel())

            for field, logits in multi_label_logits.items():
                pred = logits > 0
                gold = batch["multi_label_labels"][field].to(device) > 0
                multi_tp += int((pred & gold).sum().detach().cpu())
                multi_pred += int(pred.sum().detach().cpu())
                multi_gold += int(gold.sum().detach().cpu())

    threshold_metrics = {
        str(threshold): {
            **_prf(counts["tp"], counts["pred"], counts["gold"]),
            "predicted": counts["pred"],
            "gold": counts["gold"],
            "true_positive": counts["tp"],
        }
        for threshold, counts in entity_counts.items()
    }
    best_threshold = max(
        thresholds,
        key=lambda threshold: (
            threshold_metrics[str(threshold)]["f1"],
            threshold_metrics[str(threshold)]["recall"],
            -abs(threshold),
        ),
    )
    best_entity = threshold_metrics[str(best_threshold)]

    return {
        "entity": {
            "threshold": best_threshold,
            "precision": best_entity["precision"],
            "recall": best_entity["recall"],
            "f1": best_entity["f1"],
            "predicted": best_entity["predicted"],
            "gold": best_entity["gold"],
            "true_positive": best_entity["true_positive"],
        },
        "entity_threshold_metrics": threshold_metrics,
        "classification_accuracy": {
            field: classification_correct.get(field, 0) / max(total, 1)
            for field, total in classification_total.items()
        },
        "action_intent_micro_f1": _prf(multi_tp, multi_pred, multi_gold)["f1"],
    }


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """只移动模型需要的张量字段，文本和 offset 留在 CPU 侧。"""

    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device),
        "entity_labels": batch["entity_labels"].to(device),
        "classification_labels": {field: labels.to(device) for field, labels in batch["classification_labels"].items()},
        "multi_label_labels": {field: labels.to(device) for field, labels in batch["multi_label_labels"].items()},
    }


def _gold_entities_from_batch(batch: dict[str, Any], label_vocab: dict[str, Any]) -> list[set[tuple[str, int, int]]]:
    """从 batch 的 token span 和 offset_mapping 还原 gold 字符级实体键。"""

    entity_labels = batch["entity_labels"]
    offsets = batch["offset_mapping"]
    id_to_label = {index: label for label, index in label_vocab["entity_labels"].items()}
    gold_by_sample: list[set[tuple[str, int, int]]] = []
    for sample_index in range(entity_labels.shape[0]):
        sample_keys: set[tuple[str, int, int]] = set()
        positive = (entity_labels[sample_index] > 0).nonzero(as_tuple=False).tolist()
        for label_id, token_start, token_end in positive:
            char_start = int(offsets[sample_index, token_start, 0])
            char_end = int(offsets[sample_index, token_end, 1])
            if char_end > char_start:
                sample_keys.add((id_to_label[int(label_id)], char_start, char_end))
        gold_by_sample.append(sample_keys)
    return gold_by_sample


def _resolve_device(device_name: str, torch_module: Any) -> Any:
    """解析训练设备，auto 优先使用 CUDA。"""

    if device_name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_name)


def _prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    """计算 micro precision/recall/f1。"""

    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _parse_thresholds(raw_thresholds: str) -> list[float]:
    """解析实体预测阈值列表，并去重排序。"""

    thresholds = sorted({float(item.strip()) for item in raw_thresholds.split(",") if item.strip()})
    return thresholds or [0.0]


def _selection_score(metrics: dict[str, Any]) -> float:
    """实体 F1 持平时，用分类和多标签表现选择更有用的 checkpoint。"""

    classification_values = list(metrics.get("classification_accuracy", {}).values())
    classification_average = sum(classification_values) / len(classification_values) if classification_values else 0.0
    return (
        metrics.get("entity", {}).get("f1", 0.0) * 100.0
        + metrics.get("action_intent_micro_f1", 0.0)
        + classification_average
    )


if __name__ == "__main__":
    main()
