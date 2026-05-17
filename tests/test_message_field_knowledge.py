from __future__ import annotations

import json
from pathlib import Path

import pytest

from ev_charge_testcase_generator.message_field_knowledge import (
    MessageFieldKnowledgeLoader,
    MessageFieldKnowledgeNotFoundError,
)


def test_dc_message_field_knowledge_covers_gbt27930_messages() -> None:
    data = json.loads(Path("data/message_field_knowledge/dc_message_fields.json").read_text(encoding="utf-8"))
    messages = set(data["messages"])

    assert {
        "CHM",
        "BHM",
        "CRM",
        "BRM",
        "CTS",
        "CML",
        "BCP",
        "CRO",
        "BRO",
        "BCL",
        "BCS",
        "BSM",
        "BMT",
        "BMV",
        "BSP",
        "CCS",
        "BST",
        "CST",
        "BSD",
        "CSD",
        "BEM",
        "CEM",
    }.issubset(messages)


def test_message_field_loader_finds_field_by_alias_and_spn() -> None:
    loader = MessageFieldKnowledgeLoader("data/message_field_knowledge/dc_message_fields.json")

    protocol_version = loader.find_field("CHM", "协议版本")
    bhm_voltage = loader.find_field("BHM", "SPN2601")

    assert protocol_version is not None
    assert protocol_version.field_id == "protocol_version"
    assert protocol_version.default_value == "GBT2023"
    assert bhm_voltage is not None
    assert bhm_voltage.field_id == "max_allowable_total_voltage"
    assert bhm_voltage.unit == "V"


def test_message_field_loader_finds_field_by_semantic_key() -> None:
    loader = MessageFieldKnowledgeLoader("data/message_field_knowledge/dc_message_fields.json")

    current = loader.find_field("BCL", "demand_current")

    assert current is not None
    assert current.field_name == "电流需求"
    assert "out_of_range" in current.invalid_values


def test_message_field_loader_raises_clear_error_for_unknown_message() -> None:
    loader = MessageFieldKnowledgeLoader("data/message_field_knowledge/dc_message_fields.json")

    with pytest.raises(MessageFieldKnowledgeNotFoundError, match="未找到报文字段定义"):
        loader.get_message("XYZ")

