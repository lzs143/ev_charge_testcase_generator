# 可执行测试用例结构设计

本文档定义面向自动化测试平台执行的测试用例结构。该结构参考现有自动化用例模板中的“用例集、动作集、判据集”组织方式，但不与某一厂商模板强绑定。

## 设计目标

当前 `TestCase` 和 `TestStep` 主要用于论文原型中的结构化用例生成，能够表达测试阶段、动作、参数和预期结果。若希望自动化软件读取用例后直接执行测试，还需要进一步表达前置条件、动作编号、动作参数、判据、超时和恢复步骤。

因此新增 `ExecutableTestCase` 作为后续自动化用例生成目标结构。

## 核心结构

### ExecutableTestCase

表示一条可执行测试用例。

- `case_id`：用例编号。
- `case_name`：用例名称。
- `scene_type`：充电场景，取值如 `AC`、`DC`。
- `condition_type`：工况类型，取值如 `normal`、`fault`。
- `test_type`：测试类型，取值如 `positive`、`negative`、`compatibility`。
- `standard_source`：标准来源，如 `GB/T 34658-2025`。
- `test_stage`：测试阶段，如充电握手阶段、参数配置阶段、充电阶段、充电中止阶段。
- `target_object`：被测对象，如 `EVCC`、`SECC`、`BMS`、`充电机`。
- `preconditions`：前置条件列表。
- `steps`：自动化动作步骤列表。
- `assertions`：判据列表。
- `cleanup_steps`：测试恢复步骤列表。
- `parameters`：测试参数，如电压、电流、SOC、报文周期、超时时间。
- `fault_type`：故障类型。
- `raw_requirement`：原始自然语言需求。
- `metadata`：扩展信息。

### Precondition

表示执行前必须满足的条件。

- `condition_id`：前置条件编号。
- `description`：条件描述。
- `target`：条件作用对象。
- `parameters`：条件参数。
- `required`：是否必需。

### ExecutableStep

表示自动化平台可执行的动作。

- `step_id`：步骤序号。
- `action_id`：动作编号，可映射到动作集。
- `action_name`：动作名称。
- `action_type`：动作类型，如初始化、插枪、发送报文、等待报文、停发报文、设置参数、等待。
- `target`：动作对象。
- `parameters`：动作参数。
- `message`：涉及的报文，如 `CHM`、`BHM`、`BRM`、`BCP`。
- `signal`：涉及的信号。
- `duration_ms`：动作持续时间。
- `timeout_ms`：等待超时时间。
- `required`：是否必需。
- `description`：补充说明。

### Assertion

表示测试判定条件。

- `assertion_id`：判据编号，可映射到判据集。
- `assertion_type`：判据类型，如信号判据、报文判据、状态判据、日志判据。
- `description`：判据描述。
- `target`：判定对象。
- `signal`：信号名称。
- `message`：报文名称。
- `operator`：判断逻辑，如 `=`, `!=`, `<`, `>`, `should_send`, `should_not_send`。
- `expected_value`：期望值。
- `timeout_ms`：判定超时时间。

### CleanupStep

表示测试后的恢复动作。

- `step_id`：恢复步骤序号。
- `action_id`：动作编号。
- `action_name`：动作名称。
- `parameters`：动作参数。
- `required`：是否必需。

## 从自然语言需求中需要抽取的信息

为了生成 `ExecutableTestCase`，信息抽取目标需要从第一阶段的粗粒度字段扩展为以下字段：

- 场景信息：`scene_type`、`condition_type`、`test_type`、`standard_source`、`test_stage`。
- 对象信息：`target_object`、动作执行方、响应方。
- 前置条件：车辆状态、连接状态、协议版本、通信速率、已进入阶段。
- 动作信息：动作类型、动作名称、报文、信号、参数、持续时间。
- 故障注入信息：故障类型、触发条件、异常报文、停发报文、超时条件。
- 参数信息：电压、电流、SOC、CP占空比、CC阻值、绝缘检测电压、报文周期、超时时间。
- 判据信息：预期报文、预期信号、状态变化、判断逻辑、期望值、判定超时。
- 恢复信息：清空消息、高压复位、低压复位、拔枪、下电、释放继电器。

## 评估数据标注约定

`evaluation/eval_dataset.json` 中保留两级标注：

- `gold`：字段级抽取标准答案，用于评估当前规则抽取器的基础信息抽取能力。
- `executable_gold`：可执行测试用例标注目标，用于后续训练或评估 UIE/LLM 抽取器与自动化用例生成器。

对于无关输入文本，`executable_gold.is_relevant` 为 `false`，并通过 `reject_reason` 说明不生成测试用例。

## 示例

```json
{
  "case_id": "TC-CD-00770",
  "case_name": "DCchg兼容性-BN.1001握手阶段不发送CHM",
  "scene_type": "DC",
  "condition_type": "fault",
  "test_type": "negative",
  "standard_source": "GB/T 34658-2025",
  "test_stage": "低压辅助上电及充电握手阶段",
  "target_object": "EVCC",
  "preconditions": [
    {
      "condition_id": "PRE-001",
      "description": "测试系统和EVCC完成物理连接",
      "target": "EVCC",
      "parameters": {},
      "required": true
    }
  ],
  "steps": [
    {
      "step_id": 1,
      "action_id": "ASET-00001-0",
      "action_name": "直流插枪初始化",
      "action_type": "初始化"
    },
    {
      "step_id": 2,
      "action_id": "SLEEP-70000",
      "action_name": "等待70s",
      "action_type": "等待",
      "duration_ms": 70000,
      "description": "不发送CHM并等待握手超时"
    }
  ],
  "assertions": [
    {
      "assertion_type": "message",
      "description": "EVCC应发送BEM错误报文",
      "target": "EVCC",
      "message": "BEM",
      "operator": "should_send"
    }
  ],
  "cleanup_steps": [
    {
      "step_id": 1,
      "action_id": "AC-DC_STOP-00246",
      "action_name": "清空消息"
    }
  ]
}
```
