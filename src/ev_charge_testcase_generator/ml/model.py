"""MacBERT-GlobalPointer 多任务语义抽取模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


CLASSIFICATION_FIELDS: tuple[str, ...] = (
    "standard_source",
    "scene_type",
    "system_class",
    "test_layer",
    "condition_type",
    "test_type",
    "test_stage",
    "fault_type",
)
MULTI_LABEL_FIELDS: tuple[str, ...] = ("action_intent",)


@dataclass(frozen=True)
class SemanticModelConfig:
    """模型头部维度配置，和 label_vocab.json 保持一致。"""

    entity_type_count: int
    classification_label_counts: dict[str, int]
    multi_label_counts: dict[str, int]
    global_pointer_head_size: int = 64
    dropout: float = 0.1


class GlobalPointerHead(nn.Module):
    """GlobalPointer 实体抽取头，输出每类实体的 token span 分数。"""

    def __init__(self, hidden_size: int, entity_type_count: int, head_size: int) -> None:
        super().__init__()
        self.entity_type_count = entity_type_count
        self.head_size = head_size
        self.dense = nn.Linear(hidden_size, entity_type_count * head_size * 2)

    def forward(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """计算实体 span logits，形状为 [batch, entity_type, seq_len, seq_len]。"""

        batch_size, seq_len, _ = sequence_output.shape
        projected = self.dense(sequence_output)
        projected = projected.view(batch_size, seq_len, self.entity_type_count, self.head_size * 2)
        query, key = torch.chunk(projected, chunks=2, dim=-1)
        logits = torch.einsum("bmhd,bnhd->bhmn", query, key) / (self.head_size**0.5)

        # 只允许 start <= end 的上三角 span，避免无效反向区间参与训练和预测。
        upper_mask = torch.triu(torch.ones(seq_len, seq_len, device=sequence_output.device), diagonal=0)
        logits = logits.masked_fill(upper_mask.unsqueeze(0).unsqueeze(0) == 0, -1e12)

        if attention_mask is not None:
            token_mask = attention_mask[:, None, None, :].bool() & attention_mask[:, None, :, None].bool()
            logits = logits.masked_fill(~token_mask, -1e12)
        return logits


class MacBertGlobalPointerForSemanticExtraction(nn.Module):
    """面向充电测试需求语义抽取的 MacBERT 多任务模型。"""

    def __init__(
        self,
        model_name: str,
        config: SemanticModelConfig,
        local_files_only: bool = False,
        init_from_config_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.semantic_config = config
        if init_from_config_only:
            # 预测和错误分析会随后加载完整 checkpoint，这里只需构造结构，避免再次联网探测权重文件。
            encoder_config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.global_pointer = GlobalPointerHead(
            hidden_size=hidden_size,
            entity_type_count=config.entity_type_count,
            head_size=config.global_pointer_head_size,
        )
        self.classification_heads = nn.ModuleDict(
            {field: nn.Linear(hidden_size, count) for field, count in config.classification_label_counts.items()}
        )
        self.multi_label_heads = nn.ModuleDict(
            {field: nn.Linear(hidden_size, count) for field, count in config.multi_label_counts.items()}
        )

    @classmethod
    def from_label_vocab(
        cls,
        model_name: str,
        label_vocab: dict[str, Any],
        global_pointer_head_size: int = 64,
        dropout: float = 0.1,
        local_files_only: bool = False,
        init_from_config_only: bool = False,
    ) -> "MacBertGlobalPointerForSemanticExtraction":
        """根据标签词表创建模型，便于训练和预测脚本复用。"""

        config = SemanticModelConfig(
            entity_type_count=len(label_vocab["entity_labels"]),
            classification_label_counts={
                field: len(labels) for field, labels in label_vocab.get("classification_labels", {}).items()
            },
            multi_label_counts={field: len(labels) for field, labels in label_vocab.get("multi_label_labels", {}).items()},
            global_pointer_head_size=global_pointer_head_size,
            dropout=dropout,
        )
        return cls(
            model_name=model_name,
            config=config,
            local_files_only=local_files_only,
            init_from_config_only=init_from_config_only,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        entity_labels: torch.Tensor | None = None,
        classification_labels: dict[str, torch.Tensor] | None = None,
        multi_label_labels: dict[str, torch.Tensor] | None = None,
        entity_loss_weight: float = 1.0,
        entity_positive_weight: float = 1.0,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """前向计算，可选返回训练总 loss。"""

        encoder_kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = self.dropout(sequence_output[:, 0])

        entity_logits = self.global_pointer(sequence_output, attention_mask=attention_mask)
        classification_logits = {
            field: head(pooled_output) for field, head in self.classification_heads.items()
        }
        multi_label_logits = {field: head(pooled_output) for field, head in self.multi_label_heads.items()}

        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "entity_logits": entity_logits,
            "classification_logits": classification_logits,
            "multi_label_logits": multi_label_logits,
        }

        losses: list[torch.Tensor] = []
        if entity_labels is not None:
            entity_loss = _global_pointer_span_loss(
                entity_logits,
                entity_labels.float(),
                positive_weight=entity_positive_weight,
            )
            losses.append(entity_loss * entity_loss_weight)
        if classification_labels is not None:
            for field, logits in classification_logits.items():
                if field in classification_labels:
                    losses.append(F.cross_entropy(logits, classification_labels[field].long()))
        if multi_label_labels is not None:
            for field, logits in multi_label_logits.items():
                if field in multi_label_labels:
                    losses.append(F.binary_cross_entropy_with_logits(logits, multi_label_labels[field].float()))
        if losses:
            result["loss"] = torch.stack(losses).sum()
        return result


def _global_pointer_span_loss(logits: torch.Tensor, labels: torch.Tensor, positive_weight: float = 1.0) -> torch.Tensor:
    """GlobalPointer 多标签 span loss，对稀疏正样本加权。"""

    valid_mask = logits > -1e11
    safe_logits = torch.where(valid_mask, logits, torch.zeros_like(logits))
    raw_loss = F.binary_cross_entropy_with_logits(safe_logits, labels, reduction="none")
    positive_weights = torch.where(labels > 0, torch.full_like(labels, positive_weight), torch.ones_like(labels))
    masked_loss = raw_loss * positive_weights * valid_mask.float()
    return masked_loss.sum() / valid_mask.float().sum().clamp_min(1.0)
