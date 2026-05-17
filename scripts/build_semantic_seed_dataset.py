"""生成语义抽取种子数据集。

数据覆盖 GB/T 34658-2025 协议一致性测试、GB/T 27930-2023
直流通信报文，以及 GB/T 34657.2-2017 交直流互操作性测试中的
接口、控制导引和安全故障场景。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "semantic_dataset_seed.jsonl"
MESSAGE_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "message_field_knowledge" / "dc_message_fields.json"

PROTOCOL_STANDARD = "GB/T 34658-2025"
INTEROP_STANDARD = "GB/T 34657.2-2017"


def entity(text: str, label: str, value: str, occurrence: int = 1) -> dict[str, Any]:
    """根据实体文本自动计算 start/end。"""

    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = text.find(value, cursor)
        if start == -1:
            raise ValueError(f"{value!r} not found in {text!r}")
        cursor = start + len(value)
    return {"start": start, "end": start + len(value), "label": label, "text": value}


def sample(
    sample_id: str,
    text: str,
    entity_specs: list[tuple[str, str] | tuple[str, str, int]],
    labels: dict[str, Any],
    expected_semantics: dict[str, Any],
    source_hint: str,
) -> dict[str, Any]:
    """构造单条样本。"""

    return {
        "id": sample_id,
        "text": text,
        "entities": [entity(text, *spec) for spec in entity_specs],
        "labels": labels,
        "expected_semantics": expected_semantics,
        "source_hint": source_hint,
    }


def labels(
    standard_source: str,
    scene_type: str,
    test_layer: str,
    test_stage: str,
    fault_type: str | None,
    action_intent: list[str],
) -> dict[str, Any]:
    """构造句级分类标签。"""

    return {
        "standard_source": standard_source,
        "scene_type": scene_type,
        "system_class": "unknown",
        "test_layer": test_layer,
        "condition_type": "fault" if fault_type else "normal",
        "test_type": "negative" if fault_type else "positive",
        "test_stage": test_stage,
        "fault_type": fault_type,
        "action_intent": action_intent,
    }


def load_message_knowledge() -> dict[str, Any]:
    """读取报文字段知识库。"""

    return json.loads(MESSAGE_KNOWLEDGE_PATH.read_text(encoding="utf-8"))["messages"]


def first_field(message_data: dict[str, Any]) -> dict[str, Any]:
    """返回报文的第一个字段定义。"""

    return message_data["fields"][0]


def first_alias(field: dict[str, Any]) -> str:
    """返回字段自然语言别名。"""

    aliases = field.get("aliases", [])
    return str(aliases[0] if aliases else field["field_name"])


def add(
    rows: list[dict[str, Any]],
    text: str,
    entity_specs: list[tuple[str, str] | tuple[str, str, int]],
    label_data: dict[str, Any],
    expected_semantics: dict[str, Any],
    source_hint: str,
    prefix: str = "SEED",
) -> None:
    """追加一条样本并自动分配编号。"""

    sample_id = f"{prefix}_{len(rows) + 1:04d}"
    rows.append(sample(sample_id, text, entity_specs, label_data, expected_semantics, source_hint))


def build_protocol_rows(rows: list[dict[str, Any]], messages: dict[str, Any]) -> None:
    """构造协议一致性样本。"""

    normal_flows = [
        ("CHM", "BHM", "发送CHM握手报文后，检查BMS是否回复BHM报文"),
        ("CRM", "BRM", "发送CRM辨识报文后，等待BMS回复BRM车辆辨识报文"),
        ("CTS", "BRM", "发送CTS时间同步报文后，检查BMS是否继续发送BRM报文"),
        ("CML", "BCP", "充电参数配置阶段发送CML报文，检查车辆是否发送BCP报文"),
        ("CRO", "BRO", "发送CRO准备就绪报文，检查BMS是否回复BRO报文"),
        ("BCL", "CCS", "收到BCL电池充电需求报文后，充电机应发送CCS报文"),
        ("BCS", "CCS", "收到BCS电池充电总状态报文后，检查充电机是否发送CCS报文"),
        ("BSM", "CCS", "收到BSM车辆状态信息报文后，检查充电机是否维持CCS报文"),
        ("BMT", "CCS", "收到BMT电池温度报文后，检查充电机是否继续发送CCS报文"),
        ("BMV", "CCS", "收到BMV单体电压报文后，检查充电机是否继续发送CCS报文"),
        ("BSP", "CCS", "收到BSP电池预留状态报文后，检查充电机是否继续发送CCS报文"),
        ("BST", "CST", "车辆发送BST中止充电报文后，充电机应回复CST报文"),
        ("CST", "BSD", "充电机发送CST中止充电报文后，检查车辆是否发送BSD报文"),
        ("BSD", "CSD", "收到BSD车辆统计数据报文后，充电机应发送CSD统计报文"),
        ("BEM", "CEM", "收到BEM错误报文后，充电机应发送CEM错误报文"),
    ]
    for message, response, text in normal_flows:
        stage = messages[message]["stage"]
        specs: list[tuple[str, str] | tuple[str, str, int]] = []
        if "发送" in text:
            specs.append(("ACTION", "发送"))
        if "收到" in text:
            specs.append(("ACTION", "收到"))
        specs.append(("MESSAGE", message))
        if "检查" in text:
            specs.append(("ACTION", "检查"))
        if "等待" in text:
            specs.append(("ACTION", "等待"))
        if "回复" in text:
            specs.append(("ACTION", "回复"))
        if "BMS" in text:
            specs.append(("OBJECT", "BMS"))
        elif "车辆" in text:
            specs.append(("OBJECT", "车辆"))
        elif "充电机" in text:
            specs.append(("OBJECT", "充电机"))
        specs.append(("MESSAGE", response))
        add(
            rows,
            text,
            specs,
            labels(PROTOCOL_STANDARD, "DC", "application_layer", stage, None, ["发送报文", "等待报文", "检查响应"]),
            {"message_types": [message, response], "parameters": {}, "expected_results": [f"发送{response}报文"]},
            "GB/T 34658-2025 protocol normal flow",
        )

    message_order = list(messages)
    for message in message_order:
        data = messages[message]
        stage = data["stage"]
        response = "BEM" if data["sender"] == "SECC" else "CEM"
        target = "BMS" if response == "BEM" else "充电机"
        field = first_field(data)
        field_alias = first_alias(field)
        for fault_type, phrase, params in [
            ("报文内容错误", f"{field_alias}置为非法值", {"field_name": field_alias, "field_value": "非法值"}),
            ("报文周期错误", "周期改为500ms", {"cycle_ms": "500"}),
            ("报文ID错误", "报文ID改为0x123", {"message_id": "0x123"}),
            ("报文长度错误", "DLC长度错误", {"content_error_type": "length_error"}),
            ("报文格式错误", f"{field_alias}字段格式非法", {"field_name": field_alias, "content_error_type": "format_error"}),
        ]:
            text = f"发送{phrase}的{message}报文，检查{target}是否回复{response}错误报文"
            specs: list[tuple[str, str] | tuple[str, str, int]] = [("ACTION", "发送")]
            if "周期" in phrase:
                specs += [("PARAM_NAME", "周期"), ("PARAM_VALUE", "500ms")]
            elif "报文ID" in phrase:
                specs += [("PARAM_NAME", "报文ID"), ("PARAM_VALUE", "0x123")]
            elif "DLC" in phrase:
                specs += [("PARAM_NAME", "DLC"), ("FAULT_EXPR", "长度错误")]
            else:
                specs += [("FIELD_NAME", field_alias), ("FAULT_EXPR", "非法" if "非法" in phrase else "错误")]
            specs += [("MESSAGE", message), ("ACTION", "检查"), ("OBJECT", target), ("ACTION", "回复"), ("MESSAGE", response)]
            add(
                rows,
                text,
                specs,
                labels(PROTOCOL_STANDARD, "DC", "application_layer", stage, fault_type, ["发送报文", "设置参数", "检查响应", "故障注入"]),
                {"message_types": [message, response], "parameters": params, "expected_results": [f"发送{response}错误报文"]},
                "GB/T 27930-2023 message fault coverage",
            )

    for message in message_order:
        data = messages[message]
        stage = data["stage"]
        response = "BEM" if data["sender"] == "SECC" else "CEM"
        actor = "BMS" if data["sender"] == "SECC" else "充电机"
        for phrase, fault_type in [("超时未收到", "报文超时"), ("发送顺序错误", "报文顺序错误")]:
            text = f"{stage}{phrase}{message}报文时，{actor}应发送{response}错误报文"
            add(
                rows,
                text,
                [("TEST_CONDITION", stage), ("FAULT_EXPR", phrase), ("MESSAGE", message), ("OBJECT", actor), ("ACTION", "发送"), ("MESSAGE", response)],
                labels(PROTOCOL_STANDARD, "DC", "application_layer", stage, fault_type, ["等待报文", "检查响应", "故障注入"]),
                {"message_types": [message, response], "parameters": {}, "expected_results": [f"发送{response}错误报文"]},
                "GB/T 34658-2025 timeout and sequence coverage",
            )

    for message in message_order:
        data = messages[message]
        stage = data["stage"]
        for field in data["fields"][:2]:
            field_alias = first_alias(field)
            unit = field.get("unit") or ""
            value = "101%" if unit == "%" else "900A" if unit == "A" else "1000V" if unit == "V" else "非法值"
            fault_type = "SOC异常" if "SOC" in field_alias else "电流越界" if unit == "A" else "电压越界" if unit == "V" else "报文内容错误"
            text = f"将{message}报文{field_alias}设置为{value}，检查系统是否进入异常处理"
            add(
                rows,
                text,
                [("MESSAGE", message), ("FIELD_NAME", field_alias), ("ACTION", "设置"), ("PARAM_VALUE", value), ("OBJECT", "系统"), ("EXPECTED_EXPR", "进入异常处理")],
                labels(PROTOCOL_STANDARD, "DC", "application_layer", stage, fault_type, ["设置参数", "检查响应", "故障注入"]),
                {"message_types": [message], "parameters": {"field_name": field_alias, "field_value": value}, "expected_results": ["进入异常处理"]},
                "GB/T 27930-2023 field-level parameter coverage",
            )


def build_interop_rows(rows: list[dict[str, Any]]) -> None:
    """构造交直流互操作性样本，吸收 Excel 中的 CC/枪型/上下电/欠压等表达。"""

    ac_cases = [
        ("交流控制导引阶段将CP占空比设置为5%，车辆应停止充电", [("SIGNAL", "CP"), ("PARAM_NAME", "占空比"), ("PARAM_VALUE", "5%")], "CP信号异常", {"signal": "CP", "cp_duty": "5%"}),
        ("交流连接确认阶段CC电阻异常，检查车辆是否禁止充电", [("SIGNAL", "CC"), ("PARAM_NAME", "电阻"), ("FAULT_EXPR", "异常")], "CC信号异常", {"signal": "CC"}),
        ("整车OFF挡下设置CC阻值为1Ω并递增，检查CC插枪唤醒下限", [("OBJECT", "整车"), ("SIGNAL", "CC"), ("PARAM_NAME", "阻值"), ("PARAM_VALUE", "1Ω"), ("EXPECTED_EXPR", "CC插枪唤醒下限")], "CC信号异常", {"signal": "CC", "resistance": "1Ω"}),
        ("交流充电中CC阻值从680Ω跳变到2000Ω，车辆不应反复启停充电", [("SIGNAL", "CC"), ("PARAM_NAME", "阻值"), ("PARAM_VALUE", "680Ω"), ("PARAM_VALUE", "2000Ω"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "不应反复启停充电")], "CC信号异常", {"signal": "CC", "from": "680Ω", "to": "2000Ω"}),
        ("交流输入电压低于75V持续5分钟，车辆应识别欠压并停止充电", [("PARAM_NAME", "交流输入电压"), ("PARAM_VALUE", "75V"), ("OBJECT", "车辆"), ("FAULT_EXPR", "欠压"), ("EXPECTED_EXPR", "停止充电")], "欠压", {"voltage": "75V"}),
        ("ON挡下反复插拔交流充电枪，检查CC插入和拔出识别是否正常", [("OBJECT", "交流充电枪"), ("ACTION", "插拔"), ("SIGNAL", "CC"), ("EXPECTED_EXPR", "识别是否正常")], None, {"signal": "CC"}),
        ("使用10A单相充电枪启动交流充电，检查充电流程响应正常", [("PARAM_VALUE", "10A"), ("INTERFACE", "单相充电枪"), ("EXPECTED_EXPR", "充电流程响应正常")], None, {"gun_current": "10A"}),
        ("使用63A三相充电枪启动交流充电，检查充电流程响应正常", [("PARAM_VALUE", "63A"), ("INTERFACE", "三相充电枪"), ("EXPECTED_EXPR", "充电流程响应正常")], None, {"gun_current": "63A"}),
        ("交流充电过程中整车ON到OFF再到ON，检查充电状态是否保持正常", [("OBJECT", "整车"), ("PARAM_VALUE", "ON"), ("PARAM_VALUE", "OFF"), ("EXPECTED_EXPR", "充电状态是否保持正常")], None, {}),
        ("充电停止后不拔枪再次刷卡启动充电，车辆应正常再次启动充电", [("TEST_CONDITION", "充电停止后"), ("ACTION", "刷卡启动"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "正常再次启动充电")], None, {}),
    ]
    for text, specs, fault_type, params in ac_cases:
        add(
            rows,
            text,
            specs,
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流控制导引阶段" if "CP" in text or "占空比" in text else "交流连接确认阶段", fault_type, ["设置参数", "模拟边界", "检查响应"] if fault_type else ["检查响应"]),
            {"message_types": [], "parameters": params, "expected_results": [spec[-1] for spec in specs if spec[0] == "EXPECTED_EXPR"] or ["充电流程响应正常"]},
            "GB/T 34657.2-2017 and referenced AC Excel local cases",
        )

    for duty in ["3%", "5%", "10%", "95%", "100%"]:
        text = f"交流控制导引阶段将CP占空比设置为{duty}，检查车辆是否进入安全停充"
        add(
            rows,
            text,
            [("TEST_CONDITION", "交流控制导引阶段"), ("SIGNAL", "CP"), ("PARAM_NAME", "占空比"), ("ACTION", "设置"), ("PARAM_VALUE", duty), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "安全停充")],
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流控制导引阶段", "CP信号异常", ["设置参数", "模拟边界", "检查响应", "故障注入"]),
            {"message_types": [], "parameters": {"signal": "CP", "cp_duty": duty}, "expected_results": ["安全停充"]},
            "GB/T 34657.2-2017 AC CP boundary coverage",
        )

    for resistance in ["1Ω", "20Ω", "220Ω", "680Ω", "1500Ω", "2000Ω"]:
        text = f"交流连接确认阶段设置CC阻值为{resistance}，检查车辆对充电枪容量识别是否正确"
        add(
            rows,
            text,
            [("TEST_CONDITION", "交流连接确认阶段"), ("SIGNAL", "CC"), ("PARAM_NAME", "阻值"), ("ACTION", "设置"), ("PARAM_VALUE", resistance), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "容量识别是否正确")],
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流连接确认阶段", "CC信号异常", ["设置参数", "模拟边界", "检查响应", "故障注入"]),
            {"message_types": [], "parameters": {"signal": "CC", "resistance": resistance}, "expected_results": ["容量识别是否正确"]},
            "GB/T 34657.2-2017 and referenced Excel CC resistance coverage",
        )

    for gun, current in [("10A单相充电枪", "10A"), ("16A单相充电枪", "16A"), ("16A三相充电枪", "16A"), ("32A单相充电枪", "32A"), ("32A三相充电枪", "32A"), ("63A三相充电枪", "63A")]:
        text = f"整车OFF挡插入{gun}启动交流充电，检查充电流程响应正常"
        add(
            rows,
            text,
            [("OBJECT", "整车"), ("PARAM_VALUE", "OFF"), ("INTERFACE", gun), ("ACTION", "启动"), ("EXPECTED_EXPR", "充电流程响应正常")],
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流连接确认阶段", None, ["检查响应"]),
            {"message_types": [], "parameters": {"gun_current": current}, "expected_results": ["充电流程响应正常"]},
            "Referenced local Excel AC gun compatibility coverage",
        )

    for voltage, fault_type in [("75V", "欠压"), ("85V", "欠压"), ("260V", "过压")]:
        text = f"交流充电中输入电压调整为{voltage}，车辆应识别{fault_type}并停止充电"
        add(
            rows,
            text,
            [("PARAM_NAME", "输入电压"), ("PARAM_VALUE", voltage), ("OBJECT", "车辆"), ("FAULT_EXPR", fault_type), ("EXPECTED_EXPR", "停止充电")],
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流控制导引阶段", fault_type, ["设置参数", "模拟边界", "检查响应", "故障注入"]),
            {"message_types": [], "parameters": {"voltage": voltage}, "expected_results": ["停止充电"]},
            "Referenced local Excel AC undervoltage/overvoltage coverage",
        )

    for gear in ["ON", "OFF"]:
        text = f"整车{gear}挡反复插拔交流充电枪20次，检查CC插入拔出识别正常"
        add(
            rows,
            text,
            [("OBJECT", "整车"), ("PARAM_VALUE", gear), ("ACTION", "插拔"), ("INTERFACE", "交流充电枪"), ("SIGNAL", "CC"), ("EXPECTED_EXPR", "识别正常")],
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流连接确认阶段", None, ["检查响应"]),
            {"message_types": [], "parameters": {"gear": gear, "signal": "CC"}, "expected_results": ["识别正常"]},
            "Referenced local Excel repeated plug/unplug coverage",
        )

    for transition in ["ON到OFF再到ON", "OFF到ON再到OFF"]:
        text = f"交流充电过程中整车{transition}，检查车辆充电状态保持正常"
        add(
            rows,
            text,
            [("OBJECT", "整车"), ("PARAM_VALUE", transition), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "充电状态保持正常")],
            labels(INTEROP_STANDARD, "AC", "interoperability", "交流控制导引阶段", None, ["检查响应"]),
            {"message_types": [], "parameters": {"power_transition": transition}, "expected_results": ["充电状态保持正常"]},
            "Referenced local Excel power transition coverage",
        )

    dc_cases = [
        ("直流充电过程中断开S+通信线，车辆应停止充电并发送BEM报文", [("ACTION", "断开"), ("SIGNAL", "S+"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "停止充电"), ("ACTION", "发送"), ("MESSAGE", "BEM")], "S+断路", {"signal": "S+"}, ["BEM"]),
        ("直流充电过程中断开S-通信线，车辆应停止充电并发送BEM报文", [("ACTION", "断开"), ("SIGNAL", "S-"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "停止充电"), ("ACTION", "发送"), ("MESSAGE", "BEM")], "S-断路", {"signal": "S-"}, ["BEM"]),
        ("将S+与S-短接后，充电机应停止输出并发送CEM报文", [("SIGNAL", "S+"), ("SIGNAL", "S-"), ("FAULT_EXPR", "短接"), ("OBJECT", "充电机"), ("EXPECTED_EXPR", "停止输出"), ("ACTION", "发送"), ("MESSAGE", "CEM")], "S+S-短路", {"signal": "S+/S-"}, ["CEM"]),
        ("模拟PE断针故障，检查充电机是否停止输出并记录故障", [("ACTION", "模拟"), ("INTERFACE", "PE"), ("FAULT_EXPR", "断针故障"), ("OBJECT", "充电机"), ("EXPECTED_EXPR", "停止输出"), ("EXPECTED_EXPR", "记录故障")], "保护接地故障", {"interface": "PE"}, []),
        ("直流接口电子锁锁止失败时，车辆不应允许启动行驶", [("INTERFACE", "直流接口"), ("COMPONENT", "电子锁"), ("FAULT_EXPR", "锁止失败"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "不应允许启动行驶")], "电子锁异常", {"component": "电子锁"}, []),
        ("充电枪连接状态下车辆仍允许行驶，应判定充电与行驶互锁异常", [("TEST_CONDITION", "充电枪连接状态下"), ("OBJECT", "车辆"), ("FAULT_EXPR", "仍允许行驶"), ("EXPECTED_EXPR", "判定充电与行驶互锁异常")], "充电与行驶互锁异常", {}, []),
        ("低压辅助电源电压异常时，BMS不应进入充电握手阶段", [("COMPONENT", "低压辅助电源"), ("PARAM_NAME", "电压"), ("FAULT_EXPR", "异常"), ("OBJECT", "BMS"), ("EXPECTED_EXPR", "不应进入充电握手阶段")], "低压辅助电源异常", {"component": "低压辅助电源"}, []),
        ("绝缘检测失败后，车辆应停止充电并上报绝缘故障", [("COMPONENT", "绝缘检测"), ("FAULT_EXPR", "失败"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "停止充电"), ("FAULT_EXPR", "绝缘故障")], "绝缘故障", {"component": "绝缘检测"}, []),
        ("K5接触器状态异常时，检查车辆是否禁止充电", [("COMPONENT", "K5"), ("COMPONENT", "接触器"), ("FAULT_EXPR", "状态异常"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "禁止充电")], "K5状态异常", {"component": "K5"}, []),
        ("K6接触器无法闭合时，车辆应中止充电流程", [("COMPONENT", "K6"), ("ACTION", "闭合"), ("FAULT_EXPR", "无法闭合"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "中止充电流程")], "K6状态异常", {"component": "K6"}, []),
        ("直流充电中按下急停按钮，充电机应立即停止输出", [("FAULT_EXPR", "急停"), ("OBJECT", "充电机"), ("EXPECTED_EXPR", "停止输出")], "急停", {}, []),
        ("车辆接口断开后，BMS应发送BST中止充电报文", [("INTERFACE", "车辆接口"), ("FAULT_EXPR", "断开"), ("OBJECT", "BMS"), ("ACTION", "发送"), ("MESSAGE", "BST")], "车辆接口断开", {"interface": "车辆接口"}, ["BST"]),
    ]
    for text, specs, fault_type, params, message_types in dc_cases:
        add(
            rows,
            text,
            specs,
            labels(INTEROP_STANDARD, "DC", "interoperability", "直流控制导引阶段" if "S+" in text or "S-" in text or "低压" in text or "绝缘" in text or "K5" in text or "K6" in text else "直流接口连接确认阶段", fault_type, ["故障注入", "检查响应", "断开连接"] if "断开" in text else ["故障注入", "检查响应"]),
            {"message_types": message_types, "parameters": params, "expected_results": [spec[-1] for spec in specs if spec[0] == "EXPECTED_EXPR"]},
            "GB/T 34657.2-2017 DC interoperability coverage",
        )

    for signal, fault_type in [("S+", "S+断路"), ("S-", "S-断路"), ("CC2", "CC2信号异常"), ("PE", "PE断路")]:
        text = f"直流充电过程中断开{signal}线路，检查系统是否停止充电并记录故障"
        add(
            rows,
            text,
            [("ACTION", "断开"), ("SIGNAL" if signal != "PE" else "INTERFACE", signal), ("FAULT_EXPR", "断开"), ("OBJECT", "系统"), ("EXPECTED_EXPR", "停止充电"), ("EXPECTED_EXPR", "记录故障")],
            labels(INTEROP_STANDARD, "DC", "interoperability", "直流控制导引阶段", fault_type, ["断开连接", "检查响应", "故障注入", "记录故障"]),
            {"message_types": [], "parameters": {"signal": signal}, "expected_results": ["停止充电", "记录故障"]},
            "GB/T 34657.2-2017 DC line fault coverage",
        )

    for component, fault_type in [("K5", "K5状态异常"), ("K6", "K6状态异常"), ("电子锁", "电子锁异常")]:
        text = f"模拟{component}状态异常，检查车辆是否禁止进入充电流程"
        add(
            rows,
            text,
            [("ACTION", "模拟"), ("COMPONENT", component), ("FAULT_EXPR", "状态异常"), ("OBJECT", "车辆"), ("EXPECTED_EXPR", "禁止进入充电流程")],
            labels(INTEROP_STANDARD, "DC", "interoperability", "直流接口连接确认阶段", fault_type, ["故障注入", "检查响应"]),
            {"message_types": [], "parameters": {"component": component}, "expected_results": ["禁止进入充电流程"]},
            "GB/T 34657.2-2017 DC component state coverage",
        )

    for voltage in ["9V", "16V"]:
        text = f"低压辅助电源电压设置为{voltage}，检查BMS是否保持充电握手禁止状态"
        add(
            rows,
            text,
            [("COMPONENT", "低压辅助电源"), ("PARAM_NAME", "电压"), ("ACTION", "设置"), ("PARAM_VALUE", voltage), ("OBJECT", "BMS"), ("EXPECTED_EXPR", "充电握手禁止状态")],
            labels(INTEROP_STANDARD, "DC", "interoperability", "直流控制导引阶段", "低压辅助电源异常", ["设置参数", "模拟边界", "检查响应", "故障注入"]),
            {"message_types": [], "parameters": {"component": "低压辅助电源", "voltage": voltage}, "expected_results": ["充电握手禁止状态"]},
            "GB/T 34657.2-2017 auxiliary power coverage",
        )

    for resistance in ["50kΩ", "100kΩ", "200kΩ"]:
        text = f"绝缘检测回路模拟绝缘电阻{resistance}，车辆应判断绝缘故障并停止充电"
        add(
            rows,
            text,
            [("COMPONENT", "绝缘检测回路"), ("PARAM_NAME", "绝缘电阻"), ("PARAM_VALUE", resistance), ("OBJECT", "车辆"), ("FAULT_EXPR", "绝缘故障"), ("EXPECTED_EXPR", "停止充电")],
            labels(INTEROP_STANDARD, "DC", "interoperability", "直流控制导引阶段", "绝缘故障", ["设置参数", "模拟边界", "检查响应", "故障注入"]),
            {"message_types": [], "parameters": {"insulation_resistance": resistance}, "expected_results": ["停止充电"]},
            "GB/T 34657.2-2017 insulation coverage",
        )


def build_rows() -> list[dict[str, Any]]:
    """生成 300 条种子样本。"""

    rows: list[dict[str, Any]] = []
    messages = load_message_knowledge()
    build_protocol_rows(rows, messages)
    build_interop_rows(rows)

    base_rows = list(rows)
    variant_index = 0
    while len(rows) < 300:
        source = base_rows[variant_index % len(base_rows)]
        text = f"请测试：{source['text']}"
        rebuilt = sample(
            f"SEED_{len(rows) + 1:04d}",
            text,
            [(entity["label"], entity["text"]) for entity in source["entities"]],
            dict(source["labels"]),
            dict(source["expected_semantics"]),
            f"{source['source_hint']} | paraphrase",
        )
        rows.append(rebuilt)
        variant_index += 1

    return rows[:300]


def main() -> None:
    """写入 JSONL 文件。"""

    rows = build_rows()
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
