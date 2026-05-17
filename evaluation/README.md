# Evaluation

本目录用于评估规则版可执行语义流程。

## 数据集

- `eval_dataset.json`：包含标准样本、鲁棒样本和噪声样本。
- `gold`：用于评估基础语义抽取结果。
- `executable_gold`：用于评估动作级可执行测试用例生成与映射结果。

## 抽取评估

```bash
python evaluation/evaluate_extraction.py
```

主要指标：

- `is_relevant_accuracy`
- `scene_type_accuracy`
- `condition_type_accuracy`
- `parameters_accuracy`
- `fault_type_accuracy`
- `expected_results_accuracy`

保存结果：

```bash
python evaluation/evaluate_extraction.py --save
```

## 主流程生成评估

```bash
python evaluation/evaluate_generation.py
```

主要指标：

- `generation_success_rate`
- `check_pass_rate`
- `stage_match_accuracy`
- `average_steps`
- `average_assertions`

保存结果：

```bash
python evaluation/evaluate_generation.py --save
```

## 动作映射评估

```bash
python evaluation/evaluate_executable_generation.py
```

主要指标：

- `generation_success_rate`
- `action_mapping_rate`
- `assertion_mapping_rate`

保存结果：

```bash
python evaluation/evaluate_executable_generation.py --save
```
