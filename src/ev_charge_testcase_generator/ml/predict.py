"""MacBERT-GlobalPointer 语义抽取预测入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    """构造预测脚本命令行参数。"""

    parser = argparse.ArgumentParser(description="Predict semantic labels with a trained MacBERT-GlobalPointer model.")
    parser.add_argument("--model-dir", required=True, help="包含 best_model.pt、label_vocab.json 和 tokenizer 文件的目录")
    parser.add_argument("--text", required=True, help="待抽取的自然语言测试需求")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="实体和多标签预测阈值，默认使用 checkpoint 阈值")
    return parser


def main() -> None:
    """命令行入口。"""

    args = build_arg_parser().parse_args()

    import torch
    from transformers import AutoTokenizer

    from ev_charge_testcase_generator.ml.entity_decoder import decode_entities
    from ev_charge_testcase_generator.ml.model import MacBertGlobalPointerForSemanticExtraction

    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")
    label_vocab = json.loads((model_dir / "label_vocab.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_name = str(checkpoint.get("model_name", "hfl/chinese-macbert-base"))
    max_length = int(checkpoint.get("max_length", 256))
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

    encoded = tokenizer(
        args.text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [(int(start), int(end)) for start, end in encoded.pop("offset_mapping")[0].tolist()]
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    result = {
        "text": args.text,
        "entities": decode_entities(
            args.text,
            offsets,
            outputs["entity_logits"][0].detach().cpu(),
            label_vocab["entity_labels"],
            threshold=threshold,
            message_label_map=label_vocab.get("message_labels", {}),
        ),
        "classification_labels": _decode_classification(outputs["classification_logits"], label_vocab["classification_labels"]),
        "action_intent": _decode_multi_labels(
            outputs["multi_label_logits"].get("action_intent"),
            label_vocab.get("multi_label_labels", {}).get("action_intent", {}),
            threshold=threshold,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

def _decode_classification(logits_by_field: dict[str, Any], label_maps: dict[str, dict[str, int]]) -> dict[str, str]:
    """解码各单标签分类头。"""

    result: dict[str, str] = {}
    for field, logits in logits_by_field.items():
        id_to_label = {index: label for label, index in label_maps[field].items()}
        label_id = int(logits[0].detach().cpu().argmax().item())
        result[field] = id_to_label[label_id]
    return result


def _decode_multi_labels(logits: Any | None, label_map: dict[str, int], threshold: float) -> list[str]:
    """解码 action_intent 多标签结果。"""

    if logits is None:
        return []
    id_to_label = {index: label for label, index in label_map.items()}
    positive_ids = (logits[0].detach().cpu() > threshold).nonzero(as_tuple=False).flatten().tolist()
    return [id_to_label[int(label_id)] for label_id in positive_ids]


def _resolve_device(device_name: str, torch_module: Any) -> Any:
    """解析预测设备。"""

    if device_name == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_name)


if __name__ == "__main__":
    main()
