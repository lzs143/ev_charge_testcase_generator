from __future__ import annotations

import pytest

from ev_charge_testcase_generator.sequence_knowledge import (
    SequenceKnowledgeLoader,
    SequenceKnowledgeNotFoundError,
)


def test_sequence_knowledge_loader_loads_ac_and_dc_flows() -> None:
    loader = SequenceKnowledgeLoader("data/sequence_knowledge")

    flows = loader.load_all()

    assert {flow.flow_id for flow in flows} == {"AC_CHARGING_SEQUENCE", "DC_CHARGING_SEQUENCE"}
    assert loader.get_by_flow_id("DC_CHARGING_SEQUENCE").target_object == "BMS/EVCC"


def test_sequence_knowledge_matches_dc_chm_bhm_interaction() -> None:
    loader = SequenceKnowledgeLoader("data/sequence_knowledge")

    interaction = loader.find_interaction(
        "DC_CHARGING_SEQUENCE",
        "低压辅助上电及充电握手阶段",
        ["CHM", "BHM"],
    )

    assert interaction is not None
    assert interaction["interaction_id"] == "DC_CHM_BHM_HANDSHAKE"


def test_sequence_knowledge_matches_ac_cp_stage() -> None:
    loader = SequenceKnowledgeLoader("data/sequence_knowledge")

    stage = loader.find_stage("AC_CHARGING_SEQUENCE", "导引控制阶段", ["CP"])

    assert stage is not None
    assert stage["stage_id"] == "ac_cp_guidance"


def test_sequence_knowledge_raises_clear_error_when_missing() -> None:
    loader = SequenceKnowledgeLoader("data/sequence_knowledge")

    with pytest.raises(SequenceKnowledgeNotFoundError, match="未找到时序知识: missing"):
        loader.get_by_flow_id("missing")
