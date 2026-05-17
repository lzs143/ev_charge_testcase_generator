from __future__ import annotations

from ev_charge_testcase_generator.ml.semantic_adapter import MacBertGlobalPointerSemanticExtractor
from ev_charge_testcase_generator.semantic_extraction_factory import (
    MACBERT_GLOBALPOINTER_METHOD,
    RULE_BASED_METHOD,
    available_semantic_methods,
    build_semantic_extractor,
)
from ev_charge_testcase_generator.semantic_extractor import SemanticExtractor


def test_build_rule_based_semantic_extractor_by_default() -> None:
    extractor = build_semantic_extractor(RULE_BASED_METHOD)

    assert isinstance(extractor, SemanticExtractor)


def test_build_macbert_semantic_extractor_is_lazy() -> None:
    extractor = build_semantic_extractor(MACBERT_GLOBALPOINTER_METHOD, model_dir="outputs/macbert_globalpointer_weighted")

    assert isinstance(extractor, MacBertGlobalPointerSemanticExtractor)
    assert extractor._runtime is None


def test_available_semantic_methods_contains_gui_labels() -> None:
    methods = available_semantic_methods()

    assert methods[RULE_BASED_METHOD] == "规则抽取"
    assert methods[MACBERT_GLOBALPOINTER_METHOD] == "MacBERT-GlobalPointer 模型抽取"
