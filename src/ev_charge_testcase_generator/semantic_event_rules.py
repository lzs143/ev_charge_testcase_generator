"""测试事件语义规则库。

本模块集中维护国标充电测试需求中常见的动作、故障注入和预期反馈规则。
规则库只描述“可识别的语义模式”，具体如何构建事件由 semantic_events.py 完成。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventRule:
    """事件规则定义。"""

    name: str
    event_type: str
    action: str
    words: tuple[str, ...]
    description: str


PROTOCOL_MESSAGES: tuple[str, ...] = (
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
    "CTS",
    "BRO",
    "CRO",
    "CCS",
    "CEM",
    "BST",
    "CST",
    "BSD",
    "CSD",
)

COMMUNICATION_OBJECTS: tuple[str, ...] = (
    "BMS",
    "EVCC",
    "SECC",
    "车辆",
    "充电机",
    "充电桩",
    "系统",
)

SEND_MESSAGE_RULE = EventRule(
    name="send_message",
    event_type="stimulus",
    action="send_message",
    words=("发送", "下发", "发出", "周期发送"),
    description="测试系统发送指定协议报文。",
)

WAIT_MESSAGE_RULE = EventRule(
    name="wait_message",
    event_type="stimulus",
    action="wait_message",
    words=("等待", "接收", "收到", "监听"),
    description="测试系统等待或监听指定报文。",
)

SET_PARAMETER_RULE = EventRule(
    name="set_parameter",
    event_type="stimulus",
    action="set_parameter",
    words=("设置", "配置", "置为", "调整为", "设为"),
    description="设置报文字段、电压、电流、SOC、周期等测试参数。",
)

INJECT_FAULT_RULE = EventRule(
    name="inject_fault",
    event_type="stimulus",
    action="inject_fault",
    words=("注入", "模拟", "构造", "发送错误", "发送非法", "置为非法", "改为错误"),
    description="构造异常报文、异常信号或异常参数作为故障刺激。",
)

DISCONNECT_RULE = EventRule(
    name="disconnect",
    event_type="stimulus",
    action="disconnect",
    words=("断开", "拔出", "断开连接", "接口断开", "通信中断"),
    description="断开接口、连接或通信链路。",
)

SET_SIGNAL_RULE = EventRule(
    name="set_signal",
    event_type="stimulus",
    action="set_signal",
    words=("拉低", "拉高", "置位", "置为", "闭合", "断开", "异常"),
    description="设置 CP、CC、CC2、K 等物理或控制信号状态。",
)

REPLY_MESSAGE_RULE = EventRule(
    name="reply_message",
    event_type="expected_response",
    action="reply_message",
    words=("回复", "回发", "返回", "发送", "上报"),
    description="被测对象回复或发送指定协议报文。",
)

STOP_CHARGING_RULE = EventRule(
    name="stop_charging",
    event_type="expected_response",
    action="stop_charging",
    words=("停止充电", "停止输出", "停止供电", "切断输出", "停机"),
    description="系统进入停止充电或停止输出状态。",
)

ENTER_STATE_RULE = EventRule(
    name="enter_state",
    event_type="expected_response",
    action="enter_state",
    words=("进入", "切换到", "转入", "退出"),
    description="系统进入或退出指定流程、阶段或状态。",
)

RECORD_FAULT_RULE = EventRule(
    name="record_fault",
    event_type="expected_response",
    action="record_fault",
    words=("记录故障", "记录故障信息", "保存故障", "生成故障记录"),
    description="系统记录故障信息。",
)

SHOW_WARNING_RULE = EventRule(
    name="show_warning",
    event_type="expected_response",
    action="show_warning",
    words=("提示异常", "显示异常", "报警", "告警", "提示故障"),
    description="系统提示、显示或上报告警信息。",
)

IGNORE_MESSAGE_RULE = EventRule(
    name="ignore_message",
    event_type="expected_response",
    action="ignore_message",
    words=("忽略", "丢弃", "不处理", "不响应"),
    description="系统忽略异常报文或不对异常输入产生响应。",
)

KEEP_STATE_RULE = EventRule(
    name="keep_state",
    event_type="expected_response",
    action="keep_state",
    words=("保持", "维持", "继续"),
    description="系统保持当前状态、继续发送报文或维持输出。",
)

CHECK_RULE = EventRule(
    name="check",
    event_type="check",
    action="check",
    words=("检查", "查看", "确认", "判断", "检测", "验证"),
    description="测试系统检查报文、状态、信号或故障记录是否符合预期。",
)

STIMULUS_RULES: tuple[EventRule, ...] = (
    SEND_MESSAGE_RULE,
    WAIT_MESSAGE_RULE,
    SET_PARAMETER_RULE,
    INJECT_FAULT_RULE,
    DISCONNECT_RULE,
    SET_SIGNAL_RULE,
)

EXPECTED_RESPONSE_RULES: tuple[EventRule, ...] = (
    REPLY_MESSAGE_RULE,
    STOP_CHARGING_RULE,
    ENTER_STATE_RULE,
    RECORD_FAULT_RULE,
    SHOW_WARNING_RULE,
    IGNORE_MESSAGE_RULE,
    KEEP_STATE_RULE,
)

CHECK_RULES: tuple[EventRule, ...] = (CHECK_RULE,)

FAULT_KEYWORDS: tuple[str, ...] = (
    "错误",
    "非法",
    "无效",
    "异常",
    "超时",
    "未收到",
    "收不到",
    "中断",
    "越界",
    "DLC",
    "长度错误",
    "周期错误",
    "ID错误",
    "内容错误",
    "字段错误",
)

PARAMETER_KEYWORDS: tuple[str, ...] = (
    "电压",
    "电流",
    "SOC",
    "周期",
    "报文ID",
    "DLC",
    "字段",
    "参数",
    "占空比",
    "阻值",
)

SIGNAL_NAMES: tuple[str, ...] = ("CP", "CC", "CC2", "CC1", "K1", "K2", "K3", "K4", "K5", "K6", "S+", "S-")
