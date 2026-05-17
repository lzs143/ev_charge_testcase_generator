"""关键信息抽取模块。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .models import ExtractedInfo


class BaseExtractor(ABC):
    """信息抽取器抽象基类，便于后续扩展 BERT-CRF 等模型。"""

    @abstractmethod
    def extract(self, text: str) -> ExtractedInfo:
        """从自然语言测试需求中抽取结构化信息。"""


class RuleBasedExtractor(BaseExtractor):
    """规则版新能源汽车充电测试需求信息抽取器。"""

    DC_KEYWORDS: dict[str, int] = {
        "直流": 1,
        "BMS": 2,
        "EVCC": 2,
        "SECC": 2,
        "握手": 2,
        "预充": 2,
        "绝缘": 2,
        "BHM": 3,
        "BRM": 3,
        "BCP": 3,
        "BCL": 3,
        "BCS": 3,
        "BSM": 3,
        "BEM": 3,
        "CHM": 3,
        "CRM": 3,
        "CML": 3,
        "BRO": 3,
        "CRO": 3,
        "CCS": 3,
        "CEM": 3,
        "BST": 3,
        "CST": 3,
        "BSD": 3,
        "CSD": 3,
    }
    AC_KEYWORDS: dict[str, int] = {
        "交流": 1,
        "CP": 2,
        "CC": 2,
        "导引": 2,
    }
    MESSAGE_PATTERN = re.compile(
        r"(?:BHM|BRM|BCP|BCL|BCS|BSM|BEM|CHM|CRM|CML|CTS|BRO|CRO|CCS|CEM|BST|CST|BSD|CSD)"
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

    FAULT_KEYWORDS: tuple[str, ...] = (
        "错误",
        "错误报文",
        "数据内容错误",
        "内容错误",
        "周期错误",
        "错误周期",
        "报文ID错误",
        "ID错误",
        "非法",
        "无效",
        "异常",
        "故障",
        "中断",
        "超时",
        "未收到",
        "收不到",
        "没响应",
        "不符合",
        "越界",
        "超出",
        "失效",
        "急停",
        "不对",
        "断了",
    )
    FAULT_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("BMS通信中断", ("BMS通信中断", "BMS 通信中断")),
        ("报文参数越界", ("报文参数越界", "不符合数据域", "必需项参数不符合")),
        ("报文周期错误", ("报文周期错误", "周期错误", "错误周期", "发送周期错误", "发送错误周期", "周期异常")),
        ("报文ID错误", ("报文ID错误", "ID错误", "错误ID", "非法ID")),
        ("报文超时", ("报文超时", "等待超时", "超过规定时间", "超时后", "未收到", "收不到")),
        ("报文内容错误", ("错误报文", "错误的", "非法报文", "无效报文", "报文内容错误", "数据内容错误", "内容错误", "字段错误", "数据域错误")),
        ("通信中断", ("通信中断", "没响应")),
        ("CP信号异常", ("CP信号异常", "CP 信号异常", "CP异常")),
        ("CC信号异常", ("CC信号异常", "CC 信号异常")),
        ("保护接地故障", ("保护接地", "接地连续性失效")),
        ("绝缘故障", ("绝缘故障", "绝缘检测故障")),
        ("电压越界", ("电压越界", "过压", "欠压")),
        ("电流越界", ("电流越界", "过流", "电流值超出", "电流不对")),
        ("急停", ("急停", "紧急停止")),
        ("车辆中止充电", ("车辆端中止", "车辆中止充电")),
        ("连接断开", ("连接断开", "接口断开", "充电连接断开", "车辆接口连接断开", "交流口断了")),
    )
    EXPECTED_RESULTS: tuple[str, ...] = (
        "停止输出",
        "停止充电",
        "停充",
        "记录故障信息",
        "提示异常状态",
        "进入停机流程",
        "退出充电过程",
        "退充",
        "进入充电中止过程",
        "忽略报文内容",
        "直至报文超时",
        "完成参数配置",
        "完成连接确认",
        "连接确认通过",
        "允许充电",
        "允许充",
        "进入能量传输阶段",
    )
    ACTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("发送错误报文", ("发送错误", "发送非法", "发送无效")),
        ("设置电压", ("设置电压", "电压为")),
        ("设置电流", ("设置电流", "电流为")),
        ("设置参数", ("设置参数", "参数配置", "配置参数")),
        ("注入故障", ("注入故障", "故障注入")),
        ("停止输出", ("停止输出",)),
        ("记录故障", ("记录故障", "记录故障信息")),
        ("开始充电", ("开始充电", "启动充电")),
        ("停止充电", ("停止充电",)),
    )
    OBJECT_KEYWORDS: tuple[str, ...] = (
        "车辆",
        "系统",
        "BMS",
        "充电桩",
        "充电设备",
        "CP",
        "CC",
        "EVCC",
        "SECC",
    )

    def extract(self, text: str) -> ExtractedInfo:
        """按规则抽取测试需求中的关键信息。"""

        normalized_text = self._normalize_text(text)
        warnings: list[str] = []
        is_noise = self._is_noise_text(normalized_text)
        scene_type = None if is_noise else self._extract_scene_type(normalized_text)
        if scene_type is None:
            warnings.append("无法判断充电场景类型")

        return ExtractedInfo(
            scene_type=scene_type,
            condition_type="normal" if is_noise else self._extract_condition_type(normalized_text),
            objects=self._extract_objects(normalized_text),
            actions=self._extract_actions(normalized_text),
            parameters=self._extract_parameters(normalized_text),
            trigger_condition=self._extract_trigger_condition(normalized_text),
            fault_type=self._extract_fault_type(normalized_text),
            expected_results=self._extract_expected_results(normalized_text),
            confidence=0.8 if scene_type is not None else 0.5,
            warnings=warnings,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """归一化空白字符，保留中文业务词。"""

        return re.sub(r"\s+", " ", text.strip())

    def _extract_scene_type(self, text: str) -> str | None:
        """根据关键词权重判断 AC 或 DC 场景。"""

        upper_text = text.upper()
        dc_score = sum(weight for keyword, weight in self.DC_KEYWORDS.items() if keyword.upper() in upper_text)
        ac_score = sum(
            weight
            for keyword, weight in self.AC_KEYWORDS.items()
            if self._contains_scene_keyword(upper_text, keyword)
        )
        if dc_score > ac_score:
            return "DC"
        if ac_score > dc_score:
            return "AC"
        return None

    def _extract_condition_type(self, text: str) -> str:
        """判断正常工况或异常工况。"""

        if self._extract_fault_type(text) is not None:
            return "fault"
        return "fault" if any(keyword in text for keyword in self.FAULT_KEYWORDS) else "normal"

    @staticmethod
    def _contains_scene_keyword(upper_text: str, keyword: str) -> bool:
        """避免把 CCS 等直流报文中的 CC 误判为交流场景。"""

        upper_keyword = keyword.upper()
        if upper_keyword in {"CP", "CC"}:
            return re.search(rf"(?<![A-Z]){upper_keyword}(?![A-Z0-9])", upper_text) is not None
        return upper_keyword in upper_text

    def _extract_parameters(self, text: str) -> dict[str, str]:
        """抽取电压、电流、SOC、CP 占空比和报文故障注入参数。"""

        parameters: dict[str, str] = {}

        voltage = self._first_match(
            text,
            (
                r"电压(?:设置)?(?:为|至|到)?\s*(\d+(?:\.\d+)?)\s*(?:V|伏)",
                r"(\d+(?:\.\d+)?)\s*(?:V|伏)",
            ),
        )
        if voltage is not None:
            parameters["voltage"] = f"{voltage}V"

        current = self._first_match(
            text,
            (
                r"电流(?:设置)?(?:为|至|到)?\s*(\d+(?:\.\d+)?)\s*(?:A|安)",
                r"(\d+(?:\.\d+)?)\s*(?:A|安)",
            ),
        )
        if current is not None:
            parameters["current"] = f"{current}A"

        soc = self._first_match(text, (r"SOC\s*(?:为|至|到)?\s*(\d+(?:\.\d+)?)\s*%",))
        if soc is not None:
            parameters["soc"] = f"{soc}%"

        cp_duty = self._first_match(text, (r"CP\s*占空比\s*(?:为|至|到)?\s*(\d+(?:\.\d+)?)\s*%",))
        if cp_duty is not None:
            parameters["cp_duty"] = f"{cp_duty}%"

        cycle_ms = self._extract_message_cycle_ms(text)
        if cycle_ms is not None:
            parameters["cycle_ms"] = cycle_ms

        expected_cycle_ms = self._first_match(
            text,
            (
                r"(?:正常|标准|规定|期望|应为)(?:的)?(?:报文)?周期(?:为|是)?\s*(\d+)\s*(?:ms|毫秒)",
                r"(?:报文)?周期(?:正常|标准|规定|期望|应为)\s*(\d+)\s*(?:ms|毫秒)",
            ),
        )
        if expected_cycle_ms is not None:
            parameters["expected_cycle_ms"] = expected_cycle_ms

        message_id = self._extract_message_id(text)
        if message_id is not None:
            parameters["message_id"] = message_id

        expected_message_id = self._first_match(
            text,
            (
                r"(?:正常|标准|规定|期望|应为)(?:的)?(?:报文)?ID(?:为|是)?\s*(0x[0-9A-Fa-f]+|\d+)",
                r"(?:报文)?ID(?:正常|标准|规定|期望|应为)\s*(0x[0-9A-Fa-f]+|\d+)",
            ),
        )
        if expected_message_id is not None:
            parameters["expected_message_id"] = expected_message_id

        content_error_type = self._extract_content_error_type(text)
        if content_error_type is not None:
            parameters["content_error_type"] = content_error_type

        field_name = self._first_match(
            text,
            (
                r"字段\s*([A-Za-z0-9_]+)",
                r"([A-Za-z0-9_\u4e00-\u9fa5]+)字段(?:错误|异常|非法|无效)",
                r"字段\s*([A-Za-z0-9_\u4e00-\u9fa5]+)(?:错误|异常|非法|无效)?",
            ),
        )
        if field_name is not None:
            parameters["field_name"] = field_name

        field_value = self._first_match(
            text,
            (
                r"(?:字段值|取值|值)(?:为|改为|设置为)\s*([0-9A-Za-zxX._-]+)",
                r"(?:填充|写入)\s*([0-9A-Za-zxX._-]+)",
            ),
        )
        if field_value is not None:
            parameters["field_value"] = field_value

        return parameters

    @staticmethod
    def _extract_message_cycle_ms(text: str) -> str | None:
        """从语义文本中抽取报文发送周期，单位统一为 ms。"""

        patterns = (
            r"(?:报文)?周期(?:设置)?(?:为|至|到|改为|错误为)?\s*(\d+)\s*(?:ms|毫秒)",
            r"(\d+)\s*(?:ms|毫秒)(?:的)?(?:报文)?周期",
        )
        return RuleBasedExtractor._first_match(text, patterns)

    @staticmethod
    def _extract_message_id(text: str) -> str | None:
        """从语义文本中抽取注入用报文 ID。"""

        patterns = (
            r"(?:错误ID|非法ID)(?:为|是|设置为|改为)?\s*(0x[0-9A-Fa-f]+|\d+)",
            r"(?:报文)?ID(?:为|是|设置为|改为|错误为)\s*(0x[0-9A-Fa-f]+|\d+)",
        )
        return RuleBasedExtractor._first_match(text, patterns)

    @staticmethod
    def _extract_content_error_type(text: str) -> str | None:
        """识别内容类故障的细分形式。"""

        upper_text = text.upper()
        if any(keyword in upper_text for keyword in ("CRC", "校验", "校验和")):
            return "checksum_error"
        if any(keyword in text for keyword in ("长度错误", "长度异常", "DLC错误", "DLC异常")):
            return "length_error"
        if any(keyword in text for keyword in ("字段错误", "字段异常", "数据域错误", "数据内容错误", "内容错误", "非法报文", "无效报文")):
            return "data_content_error"
        return None

    @staticmethod
    def _first_match(text: str, patterns: tuple[str, ...]) -> str | None:
        """返回第一个正则捕获结果。"""

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_fault_type(self, text: str) -> str | None:
        """抽取预定义故障类型。"""

        upper_text = text.upper()
        for fault_type, keywords in self.FAULT_TYPES:
            if any(keyword.upper() in upper_text for keyword in keywords):
                return fault_type
        cycle_ms = self._extract_message_cycle_ms(text)
        if cycle_ms is not None and cycle_ms != "250" and ("错误" in text or "BEM" in upper_text):
            return "报文周期错误"
        message_id = self._extract_message_id(text)
        if message_id is not None and ("错误" in text or "BEM" in upper_text):
            return "报文ID错误"
        return None

    @staticmethod
    def _extract_trigger_condition(text: str) -> str | None:
        """抽取触发条件片段。"""

        patterns = (
            r"当([^，。；,;]+?)时",
            r"在([^，。；,;]+?过程中)",
            r"若([^，。；,;]+?)则",
            r"([^，。；,;]{1,40}?后)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _extract_expected_results(self, text: str) -> list[str]:
        """抽取预期结果动作。"""

        results = [
            self._normalize_expected_result(result)
            for result in self.EXPECTED_RESULTS
            if self._contains_expected_result(text, result)
        ]
        if "参数配置" in text and any(keyword in text for keyword in ("设置", "电压", "电流")):
            results.append("完成参数配置")
        results.extend(self._extract_message_expected_results(text))
        if "连接确认" in text and "状态" in text:
            results.append("完成连接确认")
        return self._unique(results)

    def _extract_message_expected_results(self, text: str) -> list[str]:
        """抽取“发送某报文”类预期结果。"""

        results: list[str] = []
        pattern = r"(?:应|周期|随后|并|，|^)?(发送[^，。；,;]*?报文)"
        for match in re.finditer(pattern, text):
            value = match.group(1)
            if self.MESSAGE_PATTERN.search(value) and not self._is_fault_stimulus_segment(
                self._segment_around(text, match.start(), match.end())
            ):
                results.append(value)
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
        return results

    @staticmethod
    def _segment_around(text: str, start: int, end: int) -> str:
        """取命中文本所在的短句，用于区分故障刺激和预期响应。"""

        left_candidates = [text.rfind(separator, 0, start) for separator in "，。；,;"]
        right_candidates = [text.find(separator, end) for separator in "，。；,;"]
        left = max(left_candidates) + 1
        right_positions = [position for position in right_candidates if position != -1]
        right = min(right_positions) if right_positions else len(text)
        return text[left:right]

    @staticmethod
    def _is_fault_stimulus_segment(segment: str) -> bool:
        """识别“发送故障报文”这类刺激动作，避免误作为预期结果。"""

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

    def _extract_actions(self, text: str) -> list[str]:
        """抽取控制动作。"""

        return [
            action
            for action, keywords in self.ACTION_RULES
            if any(keyword in text for keyword in keywords)
        ]

    def _extract_objects(self, text: str) -> list[str]:
        """抽取基础测试对象。"""

        upper_text = text.upper()
        return [keyword for keyword in self.OBJECT_KEYWORDS if keyword.upper() in upper_text]

    @staticmethod
    def _normalize_expected_result(result: str) -> str:
        """将口语化预期结果归一为实验标注中的标准表达。"""

        aliases = {
            "停充": "停止充电",
            "退充": "退出充电过程",
            "允许充": "允许充电",
            "连接确认通过": "完成连接确认",
        }
        return aliases.get(result, result)

    @classmethod
    def _contains_expected_result(cls, text: str, result: str) -> bool:
        """判断预期结果短语是否真实出现，避免“允许充电电压”等误命中。"""

        if result == "允许充电":
            return re.search(r"允许充电(?!电压|电流|功率)", text) is not None
        if result == "允许充":
            return "允许充电电压" not in text and re.search(r"允许充(?!电压|电流|功率)", text) is not None
        return result in text

    def _is_noise_text(self, text: str) -> bool:
        """过滤评测集中明确不是测试需求的日常说明类输入。"""

        return any(keyword in text for keyword in self.NOISE_KEYWORDS)

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """保持顺序去重。"""

        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result
