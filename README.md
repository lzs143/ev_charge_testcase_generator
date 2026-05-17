# ev_charge_testcase_generator

新能源汽车充电测试用例自动生成原型系统。项目用于硕士论文实验，当前阶段聚焦规则版“可执行语义流程”：把自然语言形式的交流/直流充电测试需求转换为面向自动化测试平台的动作级测试用例。

项目不实现深度学习模型、Web 服务或复杂后台任务。当前代码保留清晰模块边界，后续可以把规则抽取器替换为 BERT-CRF、UIE、LLM 或其他语义抽取模型。

## 当前主流程

```text
自然语言测试需求
-> 文本预处理
-> 有效性判断
-> 规则语义抽取
-> 可执行语义构建
-> 交/直流充电主时序知识库匹配
-> 动作集与判据映射
-> ExecutableTestCase 生成
-> JSON/Excel 输出
```

示例输入：

```text
测试直流充电过程中，BMS是否能正常回复BHM报文。
```

系统会尝试识别：

```text
场景：DC
工况：normal
被测对象：BMS/EVCC
测试系统角色：测试系统/SECC
主时序：DC_CHARGING_SEQUENCE
测试阶段：低压辅助上电及充电握手阶段
报文：BHM
```

并展开为动作级步骤、断言和清理步骤，例如：

```text
PHY_LOCK-00001
CAN_CONFIG-00002
AUX_POWER_ON-00003
CAN_RECORD_START-00004
SEND_CHM-00005
WAIT_BHM-00006
STOP_CHM-00009
CHECK_BHM_PERIOD-00007
CHECK_BHM_CONTENT-00008
AUX_POWER_OFF-00022
PHY_UNLOCK-00006
CAN_RECORD_STOP-00023
```

## 已实现功能

- 充电测试需求有效性判断，过滤会议纪要、学习笔记、售后描述等无关文本。
- AC/DC 场景识别。
- normal/fault 工况识别。
- 报文、信号、参数、故障类型、触发条件和期望结果抽取。
- 可执行语义结构 `executable_info` 构建。
- 交/直流主时序知识库加载与匹配。
- 动作集加载与动作编号映射。
- 判据编号映射。
- `ExecutableTestCase` 强类型对象生成。
- JSON/Excel 导出。
- 抽取、生成、动作映射和评测脚本的单元测试。

## 关键目录

```text
src/ev_charge_testcase_generator/
  preprocessing.py          文本预处理
  extractor.py              基础规则抽取
  semantic_extractor.py     有效性判断和扩展语义抽取
  executable_extractor.py   可执行语义构建
  sequence_knowledge.py     充电时序知识库加载
  sequence_expander.py      时序知识库展开为动作/判据/清理步骤
  action_set_loader.py      动作集加载
  action_mapper.py          动作编号映射
  assertion_mapper.py       判据编号映射
  executable_generator.py   ExecutableTestCase 生成
  exporter.py               JSON/Excel 导出
  pipeline.py               主流程编排

data/
  action_sets/              动作集
  sequence_knowledge/       交/直流充电主时序知识库
  entity_dict.json          实体词典
  samples.json              示例需求

evaluation/
  eval_dataset.json
  evaluate_extraction.py
  evaluate_generation.py
  evaluate_executable_generation.py
```

早期阶段级模板流程已经移除。后续开发应围绕 `ExecutableTestCase`、时序知识库、动作集和语义抽取链路继续演进。

## 快速使用

```python
from ev_charge_testcase_generator.pipeline import TestCaseGenerationPipeline

pipeline = TestCaseGenerationPipeline()
result = pipeline.run("测试直流充电过程中，BMS是否能正常回复BHM报文。")

print(result.test_case.to_json())
```

批量导出：

```python
from ev_charge_testcase_generator.exporter import export_cases_to_excel, export_cases_to_json
from ev_charge_testcase_generator.pipeline import TestCaseGenerationPipeline

pipeline = TestCaseGenerationPipeline()
results = pipeline.run_batch([
    "测试直流充电过程中，BMS是否能正常回复BHM报文。",
])
test_cases = [result.test_case for result in results]

export_cases_to_json(test_cases, "outputs")
export_cases_to_excel(test_cases, "outputs/test_cases.xlsx")
```

## 评测命令

```bash
python evaluation/evaluate_extraction.py
python evaluation/evaluate_generation.py
python evaluation/evaluate_executable_generation.py
```

保存结果：

```bash
python evaluation/evaluate_extraction.py --save
python evaluation/evaluate_generation.py --save
python evaluation/evaluate_executable_generation.py --save
```

## 测试命令

```bash
pytest
```

如果当前环境没有注册 `pytest` 命令，可以使用：

```bash
python -m pytest
```
