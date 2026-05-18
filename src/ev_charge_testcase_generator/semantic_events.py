"""测试需求的事件语义归一化。

实体抽取只能告诉系统“有哪些词被识别出来”，本模块进一步归一为
“谁执行什么动作、针对哪个报文、期望谁响应什么”的测试事件结构。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .semantic_event_rules import (
    CHECK_RULE,
    COMMUNICATION_OBJECTS,
    EXPECTED_RESPONSE_RULES,
    PROTOCOL_MESSAGES,
    REPLY_MESSAGE_RULE,
    SEND_MESSAGE_RULE,
    SET_PARAMETER_RULE,
    SIGNAL_NAMES,
    STIMULUS_RULES,
)
from .semantic_extractor import SemanticExtractionResult

MESSAGE_PATTERN = re.compile(rf"(?:{'|'.join(PROTOCOL_MESSAGES)})")
OBJECT_PATTERN = re.compile(rf"(?:{'|'.join(COMMUNICATION_OBJECTS)})")


@dataclass
class SemanticEvent:
    """归一化后的单个测试事件。"""

    event_type: str
    action: str
    actor: str | None = None
    target: str | None = None
    message: str | None = None
    description: str = ""
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为可展示和序列化的字典。"""

        return asdict(self)


@dataclass
class SemanticEventSet:
    """面向测试用例生成的事件语义集合。"""

    stimulus: list[SemanticEvent] = field(default_factory=list)
    expected_response: list[SemanticEvent] = field(default_factory=list)
    checks: list[SemanticEvent] = field(default_factory=list)
    communication_objects: list[str] = field(default_factory=list)
    protocol_messages: list[str] = field(default_factory=list)
    trigger_condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可展示和序列化的字典。"""

        return {
            "stimulus": [event.to_dict() for event in self.stimulus],
            "expected_response": [event.to_dict() for event in self.expected_response],
            "checks": [event.to_dict() for event in self.checks],
            "communication_objects": self.communication_objects,
            "protocol_messages": self.protocol_messages,
            "trigger_condition": self.trigger_condition,
        }


class SemanticEventBuilder:
    """基于文本模式和语义抽取结果构建测试事件。"""

    SEND_WORDS: tuple[str, ...] = SEND_MESSAGE_RULE.words
    CHECK_WORDS: tuple[str, ...] = CHECK_RULE.words
    REPLY_WORDS: tuple[str, ...] = REPLY_MESSAGE_RULE.words

    def build(self, result: SemanticExtractionResult) -> SemanticEventSet:
        """从语义抽取结果构建事件语义层。"""

        text = result.normalized_text
        protocol_messages = self._protocol_messages(text, result.message_types)
        communication_objects = self._communication_objects(text)
        stimulus = self._build_stimulus(text, result, protocol_messages)
        stimulus.extend(self._build_non_message_stimulus(text, result))
        stimulus = self._unique_events(stimulus)
        expected_response = self._build_expected_response(
            text,
            result,
            protocol_messages,
            communication_objects,
            stimulus,
        )
        expected_response.extend(self._build_state_expected_response(text, result))
        expected_response = self._unique_events(expected_response)
        checks = self._build_checks(text, expected_response)
        return SemanticEventSet(
            stimulus=stimulus,
            expected_response=expected_response,
            checks=checks,
            communication_objects=communication_objects,
            protocol_messages=protocol_messages,
            trigger_condition=self._build_trigger_condition(stimulus),
        )

    def _build_stimulus(
        self,
        text: str,
        result: SemanticExtractionResult,
        protocol_messages: list[str],
    ) -> list[SemanticEvent]:
        """识别测试系统主动施加的刺激动作。"""

        events: list[SemanticEvent] = []
        for match in re.finditer(rf"({'|'.join(self.SEND_WORDS)})\s*({MESSAGE_PATTERN.pattern})\s*报文?", text):
            action_word, message = match.group(1), match.group(2)
            events.append(
                SemanticEvent(
                    event_type="stimulus",
                    action="send_message",
                    actor=result.tester_role or "测试系统",
                    target=result.target_object,
                    message=message,
                    description=f"{result.tester_role or '测试系统'}发送{message}报文",
                    source_text=match.group(0),
                )
            )
        if not events and protocol_messages:
            message = protocol_messages[0]
            if any(word in text for word in self.SEND_WORDS):
                events.append(
                    SemanticEvent(
                        event_type="stimulus",
                        action="send_message",
                        actor=result.tester_role or "测试系统",
                        target=result.target_object,
                        message=message,
                        description=f"{result.tester_role or '测试系统'}发送{message}报文",
                        source_text=message,
                    )
                )
        return self._unique_events(events)

    def _build_non_message_stimulus(
        self,
        text: str,
        result: SemanticExtractionResult,
    ) -> list[SemanticEvent]:
        """识别参数设置、故障注入、信号控制等非报文刺激动作。"""

        events: list[SemanticEvent] = []
        for rule in STIMULUS_RULES:
            if rule.name in {"send_message", "wait_message"}:
                continue
            if not any(word in text for word in rule.words):
                continue
            if rule.name == "set_parameter" and not self._has_parameter_signal(text):
                continue
            events.append(
                SemanticEvent(
                    event_type=rule.event_type,
                    action=rule.action,
                    actor=result.tester_role or "测试系统",
                    target=result.target_object,
                    description=self._describe_non_message_stimulus(rule.action, result),
                    source_text=self._source_fragment(text, rule.words),
                )
            )
        return events

    def _build_expected_response(
        self,
        text: str,
        result: SemanticExtractionResult,
        protocol_messages: list[str],
        communication_objects: list[str],
        stimulus: list[SemanticEvent],
    ) -> list[SemanticEvent]:
        """识别被测对象应产生的响应。"""

        events: list[SemanticEvent] = []
        object_part = OBJECT_PATTERN.pattern
        reply_part = "|".join(self.REPLY_WORDS)
        pattern = rf"({object_part})?[^，。；,;]{{0,12}}?(?:是否|能否|应|正确)?[^，。；,;]{{0,8}}?({reply_part})\s*({MESSAGE_PATTERN.pattern})\s*报文?"
        for match in re.finditer(pattern, text):
            actor = match.group(1) or self._default_response_actor(result, communication_objects)
            action_word, message = match.group(2), match.group(3)
            if self._is_stimulus_message(message, stimulus):
                continue
            events.append(
                SemanticEvent(
                    event_type="expected_response",
                    action="reply_message",
                    actor=actor,
                    target=result.tester_role,
                    message=message,
                    description=f"{actor}{action_word}{message}报文",
                    source_text=match.group(0),
                )
            )
        if not events and len(protocol_messages) >= 2 and any(word in text for word in self.CHECK_WORDS):
            actor = self._default_response_actor(result, communication_objects)
            message = protocol_messages[1]
            events.append(
                SemanticEvent(
                    event_type="expected_response",
                    action="reply_message",
                    actor=actor,
                    target=result.tester_role,
                    message=message,
                    description=f"{actor}回复{message}报文",
                    source_text=message,
                )
            )
        return self._unique_events(events)

    def _build_state_expected_response(
        self,
        text: str,
        result: SemanticExtractionResult,
    ) -> list[SemanticEvent]:
        """识别停止充电、进入状态、记录故障等非报文期望反馈。"""

        events: list[SemanticEvent] = []
        for rule in EXPECTED_RESPONSE_RULES:
            if rule.name == "reply_message":
                continue
            matched_word = next((word for word in rule.words if word in text), None)
            if matched_word is None:
                continue
            description = self._state_expected_description(rule.action, matched_word)
            events.append(
                SemanticEvent(
                    event_type=rule.event_type,
                    action=rule.action,
                    actor=result.target_object or "被测对象",
                    target=result.tester_role,
                    description=description,
                    source_text=self._source_fragment(text, rule.words),
                )
            )
        return events

    def _build_checks(self, text: str, expected_response: list[SemanticEvent]) -> list[SemanticEvent]:
        """根据期望响应生成检查点事件。"""

        checks: list[SemanticEvent] = []
        for event in expected_response:
            action = "check_message_response" if event.message else f"check_{event.action}"
            description = (
                f"检查{event.actor or '被测对象'}是否正确回复{event.message}报文"
                if event.message
                else f"检查{event.description}"
            )
            checks.append(
                SemanticEvent(
                    event_type="check",
                    action=action,
                    actor="测试系统",
                    target=event.actor,
                    message=event.message,
                    description=description,
                    source_text=self._check_source_text(text) or event.source_text,
                )
            )
        return checks

    @staticmethod
    def _protocol_messages(text: str, extracted_messages: list[str]) -> list[str]:
        """过滤 BMS 等通信对象，只保留真实协议报文。"""

        values = [message for message in MESSAGE_PATTERN.findall(text)]
        values.extend(message for message in extracted_messages if message in PROTOCOL_MESSAGES)
        return _unique(values)

    @staticmethod
    def _communication_objects(text: str) -> list[str]:
        """识别 BMS、EVCC 等通信对象。"""

        return _unique(OBJECT_PATTERN.findall(text))

    @staticmethod
    def _default_response_actor(result: SemanticExtractionResult, communication_objects: list[str]) -> str:
        """推断响应动作的默认执行者。"""

        if communication_objects:
            return communication_objects[0]
        if result.target_object:
            return result.target_object
        return "被测对象"

    @staticmethod
    def _build_trigger_condition(stimulus: list[SemanticEvent]) -> str | None:
        """从刺激动作构造触发条件。"""

        if not stimulus:
            return None
        first = stimulus[0]
        if first.message:
            return f"发送{first.message}报文后"
        return first.description or None

    @staticmethod
    def _check_source_text(text: str) -> str | None:
        """截取包含检查语义的原文片段。"""

        match = re.search(r"(?:检查|查看|确认|判断|检测)[^，。；,;]*", text)
        return match.group(0) if match else None

    @staticmethod
    def _has_parameter_signal(text: str) -> bool:
        """判断文本是否包含参数或信号类对象。"""

        return any(keyword in text for keyword in SET_PARAMETER_RULE.words) and (
            any(signal in text.upper() for signal in SIGNAL_NAMES)
            or any(keyword in text for keyword in ("电压", "电流", "SOC", "周期", "字段", "参数", "阻值", "占空比"))
        )

    @staticmethod
    def _describe_non_message_stimulus(action: str, result: SemanticExtractionResult) -> str:
        """生成非报文刺激动作描述。"""

        descriptions = {
            "set_parameter": "设置测试参数",
            "inject_fault": f"注入{result.fault_type or '故障'}",
            "disconnect": "断开连接或通信链路",
            "set_signal": "设置信号状态",
        }
        return descriptions.get(action, "执行测试刺激")

    @staticmethod
    def _state_expected_description(action: str, matched_word: str) -> str:
        """生成状态类期望反馈描述。"""

        descriptions = {
            "stop_charging": matched_word,
            "enter_state": matched_word,
            "record_fault": "记录故障信息",
            "show_warning": matched_word,
            "ignore_message": matched_word,
            "keep_state": matched_word,
        }
        return descriptions.get(action, matched_word)

    @staticmethod
    def _source_fragment(text: str, words: tuple[str, ...]) -> str:
        """截取包含规则关键词的短句。"""

        positions = [text.find(word) for word in words if word in text]
        if not positions:
            return ""
        start = min(position for position in positions if position >= 0)
        left_candidates = [text.rfind(separator, 0, start) for separator in "，。；,;"]
        right_candidates = [text.find(separator, start) for separator in "，。；,;"]
        left = max(left_candidates) + 1
        right_positions = [position for position in right_candidates if position != -1]
        right = min(right_positions) if right_positions else len(text)
        return text[left:right]

    @staticmethod
    def _is_stimulus_message(message: str, events: list[SemanticEvent]) -> bool:
        """判断报文是否已经作为刺激报文使用。"""

        return any(event.message == message for event in events)

    @staticmethod
    def _unique_events(events: list[SemanticEvent]) -> list[SemanticEvent]:
        """按事件类型、动作、执行者和报文去重。"""

        result: list[SemanticEvent] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()
        for event in events:
            key = (event.event_type, event.action, event.actor, event.message)
            if key in seen:
                continue
            seen.add(key)
            result.append(event)
        return result


def build_semantic_events(result: SemanticExtractionResult) -> SemanticEventSet:
    """便捷入口：构建测试事件语义。"""

    return SemanticEventBuilder().build(result)


def _unique(values: list[str]) -> list[str]:
    """保持顺序去重。"""

    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
