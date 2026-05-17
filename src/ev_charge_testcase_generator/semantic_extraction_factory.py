"""语义抽取方式工厂。

该模块只负责选择抽取后端，规则版仍是默认路径；模型版作为可选能力接入。
"""

from __future__ import annotations

from pathlib import Path

from .semantic_extractor import SemanticExtractor


RULE_BASED_METHOD = "rule_based"
MACBERT_GLOBALPOINTER_METHOD = "macbert_globalpointer"
DEFAULT_MODEL_DIR = Path("outputs/macbert_globalpointer_weighted")


def available_semantic_methods() -> dict[str, str]:
    """返回 GUI 和命令行可以展示的语义抽取方式。"""

    return {
        RULE_BASED_METHOD: "规则抽取",
        MACBERT_GLOBALPOINTER_METHOD: "MacBERT-GlobalPointer 模型抽取",
    }


def build_semantic_extractor(
    method: str = RULE_BASED_METHOD,
    model_dir: str | Path | None = None,
    device: str = "auto",
) -> SemanticExtractor:
    """根据用户选择构造语义抽取器。"""

    if method == RULE_BASED_METHOD:
        return SemanticExtractor()
    if method == MACBERT_GLOBALPOINTER_METHOD:
        from .ml.semantic_adapter import MacBertGlobalPointerSemanticExtractor

        return MacBertGlobalPointerSemanticExtractor(
            model_dir=model_dir or DEFAULT_MODEL_DIR,
            device=device,
        )
    raise ValueError(f"未知语义抽取方式: {method}")
