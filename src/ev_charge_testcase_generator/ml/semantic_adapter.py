"""MacBERT-GlobalPointer 模型到现有语义流程的适配层。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ev_charge_testcase_generator.extractor import RuleBasedExtractor
from ev_charge_testcase_generator.models import ExtractedInfo
from ev_charge_testcase_generator.preprocessing import preprocess_text
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractionResult, SemanticExtractor


class MacBertGlobalPointerSemanticExtractor(SemanticExtractor):
    """使用训练好的 MacBERT-GlobalPointer 模型补充语义抽取结果。

    模型负责实体和分类预测，规则抽取器负责有效性兜底、参数补全和后续可执行流程兼容。
    """

    def __init__(
        self,
        model_dir: str | Path = "outputs/macbert_globalpointer_weighted",
        device: str = "auto",
        max_length: int | None = None,
        threshold: float | None = None,
        base_extractor: RuleBasedExtractor | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device_name = device
        self.max_length_override = max_length
        self.threshold_override = threshold
        self.base_extractor = base_extractor or RuleBasedExtractor()
        self.rule_extractor = SemanticExtractor(self.base_extractor)
        self._runtime: dict[str, Any] | None = None

    def extract(self, text: str) -> SemanticExtractionResult:
        """执行模型预测，并转换为现有 SemanticExtractionResult。"""

        normalized_text = preprocess_text(text)
        if not normalized_text:
            return self.rule_extractor.extract(text)

        rule_result = self.rule_extractor.extract(text)
        try:
            prediction = self._predict(normalized_text)
        except Exception as exc:
            if rule_result.is_valid:
                rule_result.warnings.append(f"模型抽取失败，已回退规则结果: {exc}")
                return rule_result
            raise RuntimeError(f"模型抽取失败: {exc}") from exc

        extracted_info = self.base_extractor.extract(normalized_text)
        classes = prediction["classification_labels"]
        entities = prediction["entities"]

        scene_type = self._pick(classes.get("scene_type"), rule_result.scene_type)
        condition_type = self._pick(classes.get("condition_type"), rule_result.condition_type) or "normal"
        test_type = self._pick(classes.get("test_type"), rule_result.test_type)
        standard_source = self._pick(classes.get("standard_source"), rule_result.standard_source)
        test_stage = self._pick(classes.get("test_stage"), rule_result.test_stage)
        # 故障类型包含强关键词和参数规则，优先保留规则判断，避免小样本分类头把“周期错误”误判为其他故障。
        fault_type = self._normalize_none(self._pick(rule_result.fault_type, classes.get("fault_type")))

        message_types = self._unique([*self._entity_texts(entities, "MESSAGE"), *rule_result.message_types])
        actions = self._unique([*self._entity_texts(entities, "ACTION"), *rule_result.actions])
        signals = self._unique([*self._entity_texts(entities, "SIGNAL"), *rule_result.signals])
        expected_results = self._unique([*rule_result.expected_results, *self._entity_texts(entities, "EXPECTED_EXPR")])
        parameters = self._merge_parameters(extracted_info, entities)

        is_valid = rule_result.is_valid or bool(scene_type and (message_types or actions or expected_results or signals))
        if not is_valid:
            return SemanticExtractionResult(
                is_valid=False,
                raw_text=text,
                normalized_text=normalized_text,
                message=rule_result.message,
                examples=rule_result.examples,
                warnings=rule_result.warnings,
            )

        scene_type = str(scene_type or "DC")
        return SemanticExtractionResult(
            is_valid=True,
            raw_text=text,
            normalized_text=normalized_text,
            message="输入有效，已使用 MacBERT-GlobalPointer 完成语义信息抽取。",
            scene_type=scene_type,
            condition_type=condition_type,
            test_type=test_type or ("negative" if condition_type == "fault" else "positive"),
            standard_source=standard_source or rule_result.standard_source,
            test_stage=test_stage,
            target_object=rule_result.target_object or self.rule_extractor._infer_target_object(),
            tester_role=rule_result.tester_role or self.rule_extractor._infer_tester_role(scene_type),
            protocol_flow=rule_result.protocol_flow or self.rule_extractor._infer_protocol_flow(scene_type),
            trigger_condition=rule_result.trigger_condition or extracted_info.trigger_condition,
            fault_type=fault_type,
            parameters=parameters,
            message_types=message_types,
            signals=signals,
            actions=actions,
            expected_results=expected_results,
            confidence=max(rule_result.confidence, 0.9),
            warnings=rule_result.warnings,
        )

    def _predict(self, text: str) -> dict[str, Any]:
        """执行单句模型推理。"""

        runtime = self._ensure_runtime()
        torch = runtime["torch"]
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        label_vocab = runtime["label_vocab"]
        device = runtime["device"]
        max_length = int(self.max_length_override or runtime["max_length"])
        threshold = float(self.threshold_override if self.threshold_override is not None else runtime["threshold"])

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

        with torch.no_grad():
            outputs = model(**encoded)

        entities = runtime["decode_entities"](
            text,
            offsets,
            outputs["entity_logits"][0].detach().cpu(),
            label_vocab["entity_labels"],
            threshold=threshold,
            message_label_map=label_vocab.get("message_labels", {}),
        )
        return {
            "entities": entities,
            "classification_labels": self._decode_classification(
                outputs["classification_logits"],
                label_vocab["classification_labels"],
            ),
            "action_intent": self._decode_multi_labels(
                outputs["multi_label_logits"].get("action_intent"),
                label_vocab.get("multi_label_labels", {}).get("action_intent", {}),
                threshold=threshold,
            ),
        }

    def _ensure_runtime(self) -> dict[str, Any]:
        """懒加载模型，避免 GUI 启动时阻塞。"""

        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoTokenizer

            from ev_charge_testcase_generator.ml.entity_decoder import decode_entities
            from ev_charge_testcase_generator.ml.model import MacBertGlobalPointerForSemanticExtraction
        except ImportError as exc:
            raise RuntimeError("请先安装 ml 依赖后再使用模型抽取方式") from exc

        checkpoint_path = self.model_dir / "best_model.pt"
        label_vocab_path = self.model_dir / "label_vocab.json"
        if not checkpoint_path.exists() or not label_vocab_path.exists():
            raise FileNotFoundError(f"未找到模型产物，请检查目录: {self.model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        label_vocab = json.loads(label_vocab_path.read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model_name = str(checkpoint.get("model_name", "hfl/chinese-macbert-base"))
        model = MacBertGlobalPointerForSemanticExtraction.from_label_vocab(
            model_name,
            label_vocab,
            local_files_only=True,
            init_from_config_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        device = self._resolve_device(self.device_name, torch)
        model.to(device)
        model.eval()

        self._runtime = {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": model,
            "label_vocab": label_vocab,
            "device": device,
            "max_length": int(checkpoint.get("max_length", 256)),
            "threshold": float(checkpoint.get("entity_threshold", 0.0)),
            "decode_entities": decode_entities,
        }
        return self._runtime

    @staticmethod
    def _decode_classification(logits_by_field: dict[str, Any], label_maps: dict[str, dict[str, int]]) -> dict[str, str]:
        """解码单标签分类结果。"""

        result: dict[str, str] = {}
        for field, logits in logits_by_field.items():
            id_to_label = {index: label for label, index in label_maps[field].items()}
            label_id = int(logits[0].detach().cpu().argmax().item())
            result[field] = id_to_label[label_id]
        return result

    @staticmethod
    def _decode_multi_labels(logits: Any | None, label_map: dict[str, int], threshold: float) -> list[str]:
        """解码 action_intent 多标签结果。"""

        if logits is None:
            return []
        id_to_label = {index: label for label, index in label_map.items()}
        positive_ids = (logits[0].detach().cpu() > threshold).nonzero(as_tuple=False).flatten().tolist()
        return [id_to_label[int(label_id)] for label_id in positive_ids]

    @staticmethod
    def _resolve_device(device_name: str, torch_module: Any) -> Any:
        """解析模型推理设备。"""

        if device_name == "auto":
            return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        return torch_module.device(device_name)

    @staticmethod
    def _entity_texts(entities: list[dict[str, Any]], label: str) -> list[str]:
        """按实体类型提取文本。"""

        return [str(entity["text"]) for entity in entities if entity.get("label") == label and entity.get("text")]

    @staticmethod
    def _merge_parameters(extracted_info: ExtractedInfo, entities: list[dict[str, Any]]) -> dict[str, str]:
        """合并规则参数和模型识别出的字段/参数实体。"""

        parameters = dict(extracted_info.parameters)
        field_names = MacBertGlobalPointerSemanticExtractor._entity_texts(entities, "FIELD_NAME")
        param_names = MacBertGlobalPointerSemanticExtractor._entity_texts(entities, "PARAM_NAME")
        param_values = MacBertGlobalPointerSemanticExtractor._entity_texts(entities, "PARAM_VALUE")
        if field_names:
            parameters.setdefault("field_name", field_names[0])
        if param_names:
            parameters.setdefault("param_name", param_names[0])
        if param_values:
            parameters.setdefault("field_value", param_values[0])
        return parameters

    @staticmethod
    def _pick(primary: str | None, fallback: str | None) -> str | None:
        """优先使用模型预测，空值时回退规则结果。"""

        return primary if primary not in {None, "", "unknown"} else fallback

    @staticmethod
    def _normalize_none(value: str | None) -> str | None:
        """将分类标签中的 None 字符串转换为空值。"""

        return None if value in {None, "", "None", "none", "null"} else value

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """保持顺序去重。"""

        result: list[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
