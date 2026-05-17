from __future__ import annotations

import pytest

from ev_charge_testcase_generator.extractor import BaseExtractor, RuleBasedExtractor


def test_rule_based_extractor_is_base_extractor() -> None:
    assert isinstance(RuleBasedExtractor(), BaseExtractor)


@pytest.mark.parametrize(
    ("text", "expected_scene_type"),
    [
        ("车辆完成握手后，设置直流充电电压为500V，电流为30A。", "DC"),
        ("直流充电过程中，执行预充并进入能量传输阶段。", "DC"),
        ("绝缘监测完成后，系统开始充电。", "DC"),
        ("交流充电过程中，当CP信号异常时，应停止充电并提示异常状态。", "AC"),
        ("导引控制完成后，交流系统开始充电。", "AC"),
        ("直流充电中检测到CP信号异常时，应停止充电。", "AC"),
    ],
)
def test_extract_scene_type(text: str, expected_scene_type: str) -> None:
    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.scene_type == expected_scene_type


@pytest.mark.parametrize(
    ("text", "expected_condition_type"),
    [
        ("直流充电过程中，当BMS通信中断时，系统应停止输出。", "fault"),
        ("交流充电过程中，当CP信号异常时，应停止充电。", "fault"),
        ("车辆完成握手后，设置直流充电电压为500V。", "normal"),
    ],
)
def test_extract_condition_type(text: str, expected_condition_type: str) -> None:
    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.condition_type == expected_condition_type


@pytest.mark.parametrize(
    ("text", "expected_parameters"),
    [
        ("设置直流充电电压为500V，电流为30A。", {"voltage": "500V", "current": "30A"}),
        ("设置直流充电电压为500 V，电流为30 A。", {"voltage": "500V", "current": "30A"}),
        ("设置电压为750伏，电流为250安。", {"voltage": "750V", "current": "250A"}),
        ("SOC为80%时，进入能量传输阶段。", {"soc": "80%"}),
        ("CP占空比为50%，导引控制应正常。", {"cp_duty": "50%"}),
        ("发送周期错误的CHM报文，周期为500ms，检查BMS回复BEM报文", {"cycle_ms": "500"}),
        (
            "发送报文ID错误的CHM报文，报文ID为0x123，检查BMS回复BEM报文",
            {"message_id": "0x123"},
        ),
        (
            "发送数据内容错误的CHM报文，字段SPN2601值改为0xFF，检查BMS回复BEM报文",
            {"content_error_type": "data_content_error", "field_name": "SPN2601", "field_value": "0xFF"},
        ),
    ],
)
def test_extract_parameters(text: str, expected_parameters: dict[str, str]) -> None:
    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.parameters == expected_parameters


@pytest.mark.parametrize(
    ("text", "expected_fault_type"),
    [
        ("直流充电过程中，当BMS通信中断时，系统应停止输出。", "BMS通信中断"),
        ("直流充电过程中，当通信中断时，系统应停止输出。", "通信中断"),
        ("发送周期错误的CHM报文，周期为500ms，检查BMS回复BEM报文", "报文周期错误"),
        ("发送错误周期CHM报文，检查BMS能否正确回复BEM报文", "报文周期错误"),
        ("发送报文ID错误的CHM报文，报文ID为0x123，检查BMS回复BEM报文", "报文ID错误"),
        ("发送数据内容错误的CHM报文，检查BMS回复BEM报文", "报文内容错误"),
        ("交流充电过程中，当CP信号异常时，应停止充电。", "CP信号异常"),
        ("交流充电过程中，当CC信号异常时，应停止充电。", "CC信号异常"),
        ("直流充电过程中，当绝缘故障时，系统应停止输出。", "绝缘故障"),
        ("直流充电过程中，当电压越界时，系统应停止输出。", "电压越界"),
        ("直流充电过程中，当电流越界时，系统应停止输出。", "电流越界"),
        ("直流充电过程中，当急停触发时，系统应停止输出。", "急停"),
        ("交流充电过程中，当连接断开时，应停止充电。", "连接断开"),
    ],
)
def test_extract_fault_type(text: str, expected_fault_type: str) -> None:
    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.fault_type == expected_fault_type


def test_extract_invalid_chm_as_fault_requirement() -> None:
    extracted_info = RuleBasedExtractor().extract("发送错误的CHM报文，检查BMS是否正常回复BEM报文")

    assert extracted_info.scene_type == "DC"
    assert extracted_info.condition_type == "fault"
    assert extracted_info.fault_type == "报文内容错误"
    assert "发送错误报文" in extracted_info.actions


@pytest.mark.parametrize(
    ("text", "expected_trigger_condition"),
        [
            ("车辆完成握手后，设置直流充电电压为500V。", "车辆完成握手后"),
            ("直流充电过程中，当BMS通信中断时，系统应停止输出。", "BMS通信中断"),
            ("在交流充电过程中，当CP信号异常时，应停止充电。", "CP信号异常"),
            ("在交流充电过程中，系统应进入能量传输阶段。", "交流充电过程中"),
            ("若连接断开则停止充电并记录故障信息。", "连接断开"),
        ],
    )
def test_extract_trigger_condition(text: str, expected_trigger_condition: str) -> None:
    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.trigger_condition == expected_trigger_condition


def test_extract_expected_results_and_actions() -> None:
    text = "直流充电过程中，当BMS通信中断时，系统应停止输出并记录故障信息。"

    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.expected_results == ["停止输出", "记录故障信息"]
    assert extracted_info.actions == ["停止输出", "记录故障"]


def test_extract_multiple_actions() -> None:
    text = "车辆完成握手后，设置直流充电电压为500V，电流为30A，并开始充电。"

    extracted_info = RuleBasedExtractor().extract(text)

    assert extracted_info.actions == ["设置电压", "设置电流", "开始充电"]


def test_unknown_scene_adds_warning() -> None:
    extracted_info = RuleBasedExtractor().extract("系统应完成参数配置。")

    assert extracted_info.scene_type is None
    assert extracted_info.condition_type == "normal"
    assert "无法判断充电场景类型" in extracted_info.warnings
    assert extracted_info.expected_results == ["完成参数配置"]
