"""测试需求有效性判断与语义信息抽取模块。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .extractor import RuleBasedExtractor
from .models import ExtractedInfo
from .preprocessing import preprocess_text


@dataclass
class SemanticExtractionResult:
    """自然语言测试需求的语义抽取结果。"""

    is_valid: bool
    raw_text: str
    normalized_text: str
    message: str
    examples: list[str] = field(default_factory=list)
    scene_type: str | None = None
    condition_type: str | None = None
    test_type: str | None = None
    standard_source: str | None = None
    test_stage: str | None = None
    target_object: str | None = None
    tester_role: str | None = None
    protocol_flow: str | None = None
    trigger_condition: str | None = None
    fault_type: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    message_types: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    expected_results: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""

        return asdict(self)


class SemanticExtractor:
    """规则版测试需求有效性判断与语义抽取器。"""

    FIXED_TARGET_OBJECT = "BMS/EVCC"

    INVALID_MESSAGE = "请输入正确的充电测试语句。"
    VALID_EXAMPLES: tuple[str, ...] = (
        "直流充电参数配置阶段，设置最高允许充电电压为750V，最大允许充电电流为250A。",
        "交流充电过程中，当CP信号异常时，应停止充电并提示异常状态。",
        "充电握手阶段，测试系统未收到CHM报文且等待超时后，EVCC应发送BEM错误报文。",
    )

    MESSAGE_PATTERN = re.compile(
        r"(?:BHM|BRM|BCP|BCL|BCS|BSM|BEM|CHM|CRM|CML|CTS|BRO|CRO|CCS|CEM|BST|CST|BSD|CSD)"
    )

    SIGNAL_KEYWORDS: tuple[str, ...] = ("CP", "CC", "CC2", "K1", "K2", "K3", "K4", "SOC")

    DOMAIN_KEYWORDS: tuple[str, ...] = (
        "充电",
        "交流",
        "直流",
        "预充",
        "充电机",
        "充电桩",
        "供电设备",
        "车辆",
        "电池",
        "BMS",
        "EVCC",
        "SECC",
        "CP",
        "CC",
        "CC2",
        "BHM",
        "BRM",
        "BCP",
        "BCS",
        "BCL",
        "BSM",
        "BEM",
        "BRO",
        "CRO",
        "CCS",
        "CEM",
        "BST",
        "CST",
        "BSD",
        "CSD",
    )

    TEST_INTENT_KEYWORDS: tuple[str, ...] = (
        "测试",
        "阶段",
        "过程",
        "当",
        "若",
        "应",
        "设置",
        "拉到",
        "发送",
        "接收",
        "未收到",
        "等待",
        "完成",
        "超时",
        "停止",
        "停充",
        "进入",
        "退出",
        "退充",
        "判断",
        "异常",
        "故障",
        "报文",
        "连接",
        "插枪",
        "占空比",
    )

    NOISE_KEYWORDS: tuple[str, ...] = (
        "会议",
        "纪要",
        "学习",
        "概念",
        "目录",
        "排版",
        "售后",
        "维修",
        "办公室",
        "物业",
        "空调",
        "人员名单",
        "不生成测试用例",
    )

    STAGE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("低压辅助上电及充电握手阶段", ("低压辅助", "握手", "BHM", "CHM")),
        ("辨识阶段", ("辨识", "BRM", "CRM")),
        ("充电准备就绪阶段", ("准备就绪", "BRO", "CRO")),
        ("充电参数配置阶段", ("参数配置", "BCP", "CML")),
        ("预充阶段", ("预充",)),
        ("充电阶段", ("充电阶段", "BCL", "BCS", "BSM", "CCS", "能量传输")),
        ("充电结束阶段", ("充电结束", "BSD", "CSD")),
        ("充电中止阶段", ("中止", "BST", "CST", "停机")),
        ("安全保护阶段", ("安全保护", "保护接地", "电流越界", "输出电流越界")),
        ("连接装置检查阶段", ("连接装置", "接口连接断开", "连接断开")),
        ("导引控制阶段", ("导引", "CP")),
        ("连接确认阶段", ("连接确认", "CC", "CC2")),
    )

    def __init__(self, base_extractor: RuleBasedExtractor | None = None) -> None:
        """初始化语义抽取器。"""

        self.base_extractor = base_extractor or RuleBasedExtractor()

    def extract(self, text: str) -> SemanticExtractionResult:
        """判断输入有效性，并抽取测试语义要素。"""

        normalized_text = preprocess_text(text)
        if not normalized_text:
            return self._invalid(text, normalized_text, ["输入文本为空"])

        extracted_info = self.base_extractor.extract(normalized_text)
        expected_results = self._merge_expected_results(
            extracted_info.expected_results,
            self._extract_protocol_expected_results(normalized_text),
        )
        message_types = self._extract_messages(normalized_text, expected_results)
        scene_type = extracted_info.scene_type or self._infer_scene_type_from_semantics(normalized_text, message_types)
        validity_warnings = self._validate_input(normalized_text, extracted_info, scene_type)
        if validity_warnings:
            return self._invalid(text, normalized_text, validity_warnings)

        scene_type = str(scene_type)
        condition_type = extracted_info.condition_type or "normal"
        return SemanticExtractionResult(
            is_valid=True,
            raw_text=text,
            normalized_text=normalized_text,
            message="输入有效，已完成测试语义信息抽取。",
            scene_type=scene_type,
            condition_type=condition_type,
            test_type="negative" if condition_type == "fault" else "positive",
            standard_source=self._infer_standard_source(scene_type, message_types),
            test_stage=self._infer_test_stage(normalized_text),
            target_object=self._infer_target_object(),
            tester_role=self._infer_tester_role(scene_type),
            protocol_flow=self._infer_protocol_flow(scene_type),
            trigger_condition=extracted_info.trigger_condition,
            fault_type=extracted_info.fault_type,
            parameters=extracted_info.parameters,
            message_types=message_types,
            signals=self._extract_signals(normalized_text),
            actions=extracted_info.actions,
            expected_results=expected_results,
            confidence=self._calculate_confidence(extracted_info, message_types),
            warnings=self._filter_warnings(extracted_info.warnings, scene_type),
        )

    def _validate_input(
        self,
        text: str,
        extracted_info: ExtractedInfo,
        scene_type: str | None,
    ) -> list[str]:
        """判断输入是否能映射为新能源汽车充电测试需求。"""

        warnings: list[str] = []
        if any(keyword in text for keyword in self.NOISE_KEYWORDS):
            warnings.append("输入更像无关说明或日常文本")

        has_domain = any(keyword.upper() in text.upper() for keyword in self.DOMAIN_KEYWORDS)
        has_intent = any(keyword in text for keyword in self.TEST_INTENT_KEYWORDS)
        has_structured_signal = bool(extracted_info.parameters or extracted_info.fault_type or extracted_info.expected_results)

        if scene_type is None:
            warnings.append("未识别到交流或直流充电场景")
        if not has_domain:
            warnings.append("缺少新能源汽车充电相关对象或术语")
        if not has_intent and not has_structured_signal:
            warnings.append("缺少测试动作、触发条件或预期结果")
        return warnings

    def _infer_scene_type_from_semantics(self, text: str, messages: list[str]) -> str | None:
        """根据报文和阶段语义补充判断 AC/DC 场景。"""

        dc_messages = {
            "EVCC",
            "SECC",
            "BMS",
            "BHM",
            "BRM",
            "BCP",
            "BCL",
            "BCS",
            "BSM",
            "BEM",
            "CHM",
            "CRM",
            "CML",
            "BRO",
            "CRO",
            "CCS",
            "CEM",
            "BST",
            "CST",
            "BSD",
            "CSD",
        }
        if set(messages) & dc_messages:
            return "DC"
        if any(keyword in text for keyword in ("握手", "辨识", "参数配置", "充电中止")):
            return "DC"
        if "CP" in text or "导引" in text:
            return "AC"
        return None

    def _extract_protocol_expected_results(self, text: str) -> list[str]:
        """抽取协议报文类预期结果。"""

        results: list[str] = []
        patterns = (
            r"(?:应|周期|随后|并|，|^)(发送[^，。；,;]*?报文)",
            r"应(停止充电)",
            r"应(停止输出)",
            r"应(退出充电过程)",
            r"应(进入[^，。；,;]*?流程)",
            r"应(进入[^，。；,;]*?阶段)",
            r"应(提示异常状态)",
            r"应(记录故障信息)",
            r"应(忽略报文内容)",
            r"(直至报文超时)",
            r"(完成连接确认)",
            r"(连接确认通过)",
            r"(允许充电)",
            r"(允许充)",
            r"(停充)",
            r"(退充)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if not self._is_fault_stimulus_segment(self._segment_around(text, match.start(), match.end())):
                    results.append(self._normalize_expected_result(match.group(1)))
        message_pattern = (
            r"((?:BHM|BRM|BCP|BCL|BCS|BSM|BEM|CHM|CRM|CML|BRO|CRO|CCS|CEM|BST|CST|BSD|CSD)"
            r"[^，。；,;、和]*?报文)"
        )
        for match in re.finditer(message_pattern, text):
            value = match.group(1)
            if self._is_fault_stimulus_segment(self._segment_around(text, match.start(), match.end())):
                continue
            if not value.startswith("发送"):
                value = f"发送{value}"
            results.append(value)
        if "报错" in text:
            results.append("发送错误报文")
        return self._unique(results)

    @staticmethod
    def _segment_around(text: str, start: int, end: int) -> str:
        """取命中文本所在短句，避免把故障刺激误判为预期结果。"""

        left_candidates = [text.rfind(separator, 0, start) for separator in "，。；,;"]
        right_candidates = [text.find(separator, end) for separator in "，。；,;"]
        left = max(left_candidates) + 1
        right_positions = [position for position in right_candidates if position != -1]
        right = min(right_positions) if right_positions else len(text)
        return text[left:right]

    @staticmethod
    def _is_fault_stimulus_segment(segment: str) -> bool:
        """判断短句是否描述发送故障报文的刺激动作。"""

        if "发送" not in segment:
            return False
        if any(keyword in segment for keyword in ("应", "回复", "回发", "检查", "能否", "是否")):
            return False
        fault_keywords = (
            "错误周期",
            "周期错误",
            "报文ID错误",
            "ID错误",
            "错误ID",
            "非法ID",
            "数据内容错误",
            "内容错误",
            "字段错误",
            "非法",
            "无效",
            "错误的",
            "错误报文",
        )
        return any(keyword in segment for keyword in fault_keywords)

    def _merge_expected_results(self, base_results: list[str], extra_results: list[str]) -> list[str]:
        """合并基础预期结果和协议类预期结果。"""

        return self._unique([*base_results, *extra_results])

    def _invalid(self, raw_text: str, normalized_text: str, warnings: list[str]) -> SemanticExtractionResult:
        """构造无效输入结果。"""

        return SemanticExtractionResult(
            is_valid=False,
            raw_text=raw_text,
            normalized_text=normalized_text,
            message=self.INVALID_MESSAGE,
            examples=list(self.VALID_EXAMPLES),
            warnings=warnings,
        )

    def _extract_messages(self, text: str, expected_results: list[str]) -> list[str]:
        """抽取报文和通信对象缩略语。"""

        values = self.MESSAGE_PATTERN.findall(text)
        for expected_result in expected_results:
            values.extend(self.MESSAGE_PATTERN.findall(expected_result))
        return self._unique(values)

    def _extract_signals(self, text: str) -> list[str]:
        """抽取物理信号或关键测试信号。"""

        upper_text = text.upper()
        signals: list[str] = []
        for signal in self.SIGNAL_KEYWORDS:
            if signal in {"CP", "CC"}:
                if re.search(rf"(?<![A-Z]){signal}(?![A-Z0-9])", upper_text):
                    signals.append(signal)
                continue
            if signal.upper() in upper_text:
                signals.append(signal)
        return signals

    def _infer_test_stage(self, text: str) -> str | None:
        """根据阶段关键词推断测试阶段。"""

        upper_text = text.upper()
        if "BRM" in upper_text and any(keyword in text for keyword in ("未收到", "收不到", "超时", "超过规定时间")):
            return "低压辅助上电及充电握手阶段"
        if "导引控制完成" in text:
            return "导引控制阶段"
        if "BCL" in upper_text and any(keyword in text for keyword in ("不对", "退充")):
            return "充电中止阶段"
        if "BMS" in upper_text and any(keyword in text for keyword in ("没响应", "通信中断")):
            return "充电阶段"
        if any(keyword in text for keyword in ("交流口断了", "保护接地", "输出电流越界")):
            return "安全保护阶段"
        for stage, keywords in self.STAGE_RULES:
            if any(keyword.upper() in upper_text for keyword in keywords):
                return stage
        return None

    def _infer_target_object(self) -> str:
        """返回固定被测对象。

        本原型面向车端 BMS/EVCC 测试，测试系统在交流或直流场景中模拟供电侧。
        """

        return self.FIXED_TARGET_OBJECT

    @staticmethod
    def _infer_tester_role(scene_type: str) -> str:
        """根据充电场景推断测试系统扮演的协议角色。"""

        if scene_type == "DC":
            return "测试系统/SECC"
        return "测试系统/交流供电设备"

    @staticmethod
    def _infer_protocol_flow(scene_type: str) -> str:
        """根据充电场景选择主时序模板。"""

        if scene_type == "DC":
            return "DC_CHARGING_SEQUENCE"
        return "AC_CHARGING_SEQUENCE"

    @staticmethod
    def _infer_standard_source(scene_type: str, messages: list[str]) -> str | None:
        """根据场景和报文推断标准来源。"""

        if scene_type == "DC" and messages:
            return "GB/T 34658-2025"
        if scene_type == "DC":
            return "GB/T 27930-2023"
        if scene_type == "AC":
            return "GB/T 18487.1-2023"
        return None

    @staticmethod
    def _calculate_confidence(extracted_info: ExtractedInfo, messages: list[str]) -> float:
        """根据识别到的语义要素给出规则置信度。"""

        score = extracted_info.confidence
        if extracted_info.parameters:
            score += 0.05
        if extracted_info.fault_type:
            score += 0.05
        if extracted_info.expected_results:
            score += 0.05
        if messages:
            score += 0.05
        return min(score, 0.95)

    @staticmethod
    def _filter_warnings(warnings: list[str], scene_type: str | None) -> list[str]:
        """过滤已经由语义规则补偿解决的告警。"""

        if scene_type is None:
            return warnings
        return [warning for warning in warnings if warning != "无法判断充电场景类型"]

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """保持顺序去重。"""

        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result

    @staticmethod
    def _normalize_expected_result(result: str) -> str:
        """归一口语化预期结果，便于和动作/判据映射表对齐。"""

        aliases = {
            "停充": "停止充电",
            "退充": "退出充电过程",
            "允许充": "允许充电",
            "连接确认通过": "完成连接确认",
        }
        return aliases.get(result, result)
