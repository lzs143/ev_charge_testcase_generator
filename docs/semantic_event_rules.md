# 测试事件语义规则库说明

本文档说明 `src/ev_charge_testcase_generator/semantic_event_rules.py` 中维护的事件语义规则。该规则库用于把实体抽取结果进一步归一为测试用例生成所需的事件结构。

## 1. 设计目的

实体抽取只能识别出文本中出现的动作词、报文名、对象名和参数值，例如：

```text
发送CHM报文，查看BMS是否正确回复BHM报文
```

基础抽取可能得到：

```json
{
  "actions": ["发送", "查看", "回复"],
  "message_types": ["CHM", "BHM"],
  "objects": ["BMS"]
}
```

但完整测试用例生成还需要理解事件关系：

```text
测试刺激：测试系统发送 CHM 报文
期望响应：BMS 回复 BHM 报文
检查点：检查 BMS 是否正确回复 BHM 报文
```

因此新增事件语义层，将自然语言需求归一为：

- `stimulus`：测试系统施加的刺激动作。
- `expected_response`：被测对象应产生的反馈。
- `checks`：测试系统需要验证的判据。

## 2. 协议报文与通信对象

规则库首先区分“协议报文”和“通信对象”，避免把 `BMS` 误识别为报文。

### 2.1 协议报文

当前维护的直流充电协议报文包括：

| 报文 | 说明 |
| --- | --- |
| CHM | 充电机握手报文 |
| BHM | BMS 握手报文 |
| CRM | 充电机辨识报文 |
| BRM | BMS/车辆辨识报文 |
| CML | 充电机最大输出能力报文 |
| BCP | 动力蓄电池充电参数报文 |
| CRO | 充电机准备就绪报文 |
| BRO | BMS 准备就绪报文 |
| BCL | 电池充电需求报文 |
| BCS | 电池充电总状态报文 |
| BSM | 动力蓄电池状态信息报文 |
| CCS | 充电机充电状态报文 |
| BST | BMS 中止充电报文 |
| CST | 充电机中止充电报文 |
| BSD | BMS 统计数据报文 |
| CSD | 充电机统计数据报文 |
| BEM | BMS 错误报文 |
| CEM | 充电机错误报文 |
| CTS | 时间同步报文 |

### 2.2 通信对象

当前维护的通信对象包括：

```text
BMS、EVCC、SECC、车辆、充电机、充电桩、系统
```

这些对象不会进入 `protocol_messages`，而会进入 `communication_objects`。

## 3. 刺激动作规则

`stimulus` 表示测试系统主动施加给被测对象或环境的动作。

| 规则名 | 归一化 action | 关键词 | 语义说明 |
| --- | --- | --- | --- |
| `send_message` | `send_message` | 发送、下发、发出、周期发送 | 测试系统发送指定协议报文 |
| `wait_message` | `wait_message` | 等待、接收、收到、监听 | 测试系统等待或监听报文 |
| `set_parameter` | `set_parameter` | 设置、配置、置为、调整为、设为 | 设置字段、电压、电流、SOC、周期等参数 |
| `inject_fault` | `inject_fault` | 注入、模拟、构造、发送错误、发送非法、置为非法、改为错误 | 构造异常报文、异常信号或异常参数 |
| `disconnect` | `disconnect` | 断开、拔出、断开连接、接口断开、通信中断 | 断开接口、连接或通信链路 |
| `set_signal` | `set_signal` | 拉低、拉高、置位、置为、闭合、断开、异常 | 设置 CP、CC、CC2、K 等信号状态 |

示例：

```text
发送CHM报文
```

归一化为：

```json
{
  "event_type": "stimulus",
  "action": "send_message",
  "actor": "测试系统/SECC",
  "message": "CHM"
}
```

## 4. 期望反馈规则

`expected_response` 表示被测对象或系统应产生的响应。

| 规则名 | 归一化 action | 关键词 | 语义说明 |
| --- | --- | --- | --- |
| `reply_message` | `reply_message` | 回复、回发、返回、发送、上报 | 被测对象回复或发送指定协议报文 |
| `stop_charging` | `stop_charging` | 停止充电、停止输出、停止供电、切断输出、停机 | 系统停止充电或停止输出 |
| `enter_state` | `enter_state` | 进入、切换到、转入、退出 | 系统进入或退出指定流程、阶段或状态 |
| `record_fault` | `record_fault` | 记录故障、记录故障信息、保存故障、生成故障记录 | 系统记录故障信息 |
| `show_warning` | `show_warning` | 提示异常、显示异常、报警、告警、提示故障 | 系统提示或显示异常状态 |
| `ignore_message` | `ignore_message` | 忽略、丢弃、不处理、不响应 | 系统忽略异常报文或不响应 |
| `keep_state` | `keep_state` | 保持、维持、继续 | 系统保持当前状态或继续执行原有行为 |

示例：

```text
查看BMS是否正确回复BHM报文
```

归一化为：

```json
{
  "event_type": "expected_response",
  "action": "reply_message",
  "actor": "BMS",
  "message": "BHM"
}
```

## 5. 检查点规则

`checks` 表示测试系统需要验证的判据。当前检查点主要由期望反馈自动生成。

| 来源 | 检查 action | 示例 |
| --- | --- | --- |
| `reply_message` | `check_message_response` | 检查 BMS 是否正确回复 BHM 报文 |
| `stop_charging` | `check_stop_charging` | 检查系统是否停止充电 |
| `enter_state` | `check_enter_state` | 检查系统是否进入异常处理流程 |
| `record_fault` | `check_record_fault` | 检查系统是否记录故障信息 |
| `show_warning` | `check_show_warning` | 检查系统是否提示异常 |
| `ignore_message` | `check_ignore_message` | 检查系统是否忽略异常报文 |
| `keep_state` | `check_keep_state` | 检查系统是否保持原状态 |

## 6. 故障、参数与信号关键词

规则库额外维护三类关键词，供后续扩展使用。

### 6.1 故障关键词

```text
错误、非法、无效、异常、超时、未收到、收不到、中断、越界、DLC、长度错误、周期错误、ID错误、内容错误、字段错误
```

### 6.2 参数关键词

```text
电压、电流、SOC、周期、报文ID、DLC、字段、参数、占空比、阻值
```

### 6.3 信号名称

```text
CP、CC、CC2、CC1、K1、K2、K3、K4、K5、K6、S+、S-
```

## 7. 当前泛化能力与边界

当前规则库可以覆盖以下常见需求：

1. 报文发送与响应类：
   - 发送 CHM，检查 BMS 回复 BHM。
   - 发送 CRM，等待 BMS 回复 BRM。
2. 报文异常类：
   - 发送周期错误的 CHM 报文。
   - 发送报文 ID 错误的 CRM 报文。
   - 发送 DLC 长度错误的 CRO 报文。
3. 参数设置类：
   - 设置电压为 500V。
   - 设置电流为 30A。
   - 将某字段置为非法值。
4. 状态反馈类：
   - 系统停止充电。
   - 进入异常处理流程。
   - 记录故障信息。
   - 提示异常状态。
5. 信号与连接类：
   - CP 信号异常。
   - CC 阻值异常。
   - 断开连接或接口断开。

当前仍然存在的边界：

1. 对复杂长句的多事件切分还比较粗糙。
2. 对“谁发送、谁接收”的角色判断仍依赖固定场景假设。
3. 对字段名和参数值的归一化仍主要依赖规则抽取器。
4. 对国标全部条款的覆盖还需要继续补充测试样例和规则映射。

## 8. 后续扩展方式

新增规则时优先修改：

```text
src/ev_charge_testcase_generator/semantic_event_rules.py
```

如果只是新增动作词或反馈词，通常只需要扩展对应 `EventRule.words`。

如果需要新增事件类型，应同步修改：

```text
src/ev_charge_testcase_generator/semantic_events.py
```

并补充测试：

```text
tests/test_semantic_events.py
```

推荐新增规则时同时提供至少一个自然语言样例，确保规则不会破坏已有的“发送刺激/期望响应/检查点”分离逻辑。
