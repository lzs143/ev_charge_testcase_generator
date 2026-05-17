"""自动化判据语义到判据编号的映射模块。"""

from __future__ import annotations

from .models import Assertion


class AssertionMapper:
    """将判据语义映射为自动化判据编号。"""

    MESSAGE_ASSERTIONS: dict[str, tuple[str, str]] = {
        "BEM": ("PJ-BEM-001", "检测到BEM错误报文"),
        "CEM": ("PJ-CEM-001", "检测到CEM错误报文"),
        "BST": ("PJ-BST-001", "检测到BST车辆中止充电报文"),
        "CST": ("PJ-CST-001", "检测到CST充电机中止充电报文"),
        "BHM": ("PJ-BHM-001", "检测到BHM车辆握手报文"),
        "BRM": ("PJ-BRM-001", "检测到BRM车辆辨识报文"),
        "BCP": ("PJ-BCP-001", "检测到BCP车辆充电参数报文"),
        "BCL": ("PJ-BCL-001", "检测到BCL电池充电需求报文"),
        "BCS": ("PJ-BCS-001", "检测到BCS电池充电总状态报文"),
        "BSM": ("PJ-BSM-001", "检测到BSM车辆状态信息报文"),
        "BRO": ("PJ-BRO-001", "检测到BRO车辆充电准备就绪报文"),
        "CRO": ("PJ-CRO-001", "检测到CRO充电机输出准备就绪报文"),
        "CCS": ("PJ-CCS-001", "检测到CCS充电机充电状态报文"),
        "CML": ("PJ-CML-001", "检测到CML充电机最大输出能力报文"),
        "BSD": ("PJ-BSD-001", "检测到BSD车辆统计数据报文"),
        "CSD": ("PJ-CSD-001", "检测到CSD充电机统计数据报文"),
    }

    STATE_ASSERTIONS: tuple[tuple[str, str, str], ...] = (
        ("停止输出", "PJ-STOP-OUTPUT", "系统停止输出"),
        ("停止充电", "PJ-STOP-CHARGE", "系统停止充电"),
        ("退出充电过程", "PJ-EXIT-CHARGE", "系统退出充电过程"),
        ("退充", "PJ-EXIT-CHARGE", "系统退出充电过程"),
        ("进入充电中止过程", "PJ-STOPPING-PROCESS", "系统进入充电中止过程"),
        ("进入停机流程", "PJ-SHUTDOWN-PROCESS", "系统进入停机流程"),
        ("进入能量传输阶段", "PJ-ENERGY-TRANSFER", "系统进入能量传输阶段"),
        ("允许充电", "PJ-ALLOW-CHARGE", "车辆允许充电"),
        ("允许充", "PJ-ALLOW-CHARGE", "车辆允许充电"),
        ("完成连接确认", "PJ-CONNECTION-CONFIRM", "完成连接确认"),
        ("连接确认通过", "PJ-CONNECTION-CONFIRM", "完成连接确认"),
        ("完成参数配置", "PJ-PARAM-CONFIG", "完成参数配置"),
        ("提示异常状态", "PJ-FAULT-PROMPT", "提示异常状态"),
        ("记录故障信息", "PJ-FAULT-RECORD", "记录故障信息"),
        ("忽略报文内容", "PJ-IGNORE-MESSAGE", "忽略异常报文内容"),
        ("直至报文超时", "PJ-MESSAGE-TIMEOUT", "等待至报文超时"),
        ("发送错误报文", "PJ-ERROR-MESSAGE", "发送错误报文"),
    )

    def map_assertion(self, assertion: Assertion) -> Assertion:
        """返回填充判据编号后的判据。"""

        if assertion.assertion_id:
            return assertion

        message = self._first_message(assertion.message)
        if message in self.MESSAGE_ASSERTIONS:
            assertion.assertion_id, mapped_description = self.MESSAGE_ASSERTIONS[message]
            if not assertion.description:
                assertion.description = mapped_description
            return assertion

        for keyword, assertion_id, mapped_description in self.STATE_ASSERTIONS:
            if keyword in assertion.description or keyword == assertion.expected_value:
                assertion.assertion_id = assertion_id
                if not assertion.description:
                    assertion.description = mapped_description
                return assertion

        assertion.assertion_id = "PJ-UNMAPPED"
        return assertion

    @staticmethod
    def _first_message(message_text: str | None) -> str | None:
        """从报文列表中取第一个用于判定的报文。"""

        if not message_text:
            return None
        messages = [message.strip() for message in message_text.replace("，", ",").split(",") if message.strip()]
        for message in messages:
            if message not in {"EVCC", "SECC", "BMS"}:
                return message
        return messages[0] if messages else None
