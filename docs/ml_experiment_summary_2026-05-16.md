# MacBERT-GlobalPointer 多任务语义抽取实验总结（2026-05-16）

## 1. 实验背景

本实验面向“新能源汽车充电测试用例自动生成”任务，目标是将自然语言测试需求解析为后续规则版流程可以消费的结构化语义信息。当前规则版原型已经具备文本预处理、规则抽取、模板匹配、规则校验和 JSON/Excel 输出能力，本实验在不破坏原有规则流程的前提下，补充基于 MacBERT-GlobalPointer 的语义抽取模型，用于验证深度学习方法在测试需求语义理解环节中的可行性。

实验任务被建模为多任务学习问题，模型同时完成三类预测：

1. 实体抽取：识别测试需求中的动作、报文、字段、参数值、故障表达、期望表达等片段。
2. 单标签分类：判断标准来源、测试场景、测试层级、条件类型、测试类型、测试阶段、故障类型等全句级语义类别。
3. 多标签分类：判断测试需求包含的动作意图，例如发送报文、设置参数、故障注入、检查响应等。

本次实验重点关注两个问题：

1. MacBERT 编码器结合 GlobalPointer 是否能够在小规模语义标注数据上完成实体边界识别。
2. 领域后处理是否能够缓解小数据场景下实体 span 过长、边界不稳定的问题。

## 2. 数据与标签体系

实验数据来自项目内语义训练数据集，数据划分文件位于：

- `data/semantic_dataset/train.jsonl`
- `data/semantic_dataset/dev.jsonl`
- `data/semantic_dataset/test.jsonl`
- `data/semantic_dataset/label_vocab.json`

数据规模如下：

| 数据集 | 样本数 | 用途 |
| --- | ---: | --- |
| train | 240 | 模型参数训练 |
| dev | 30 | 训练过程评估、模型选择、阈值观察 |
| test | 30 | 最终泛化效果评估 |
| total | 300 | 全部语义标注样本 |

实体标签共 12 类：

| 标签 | 业务含义 |
| --- | --- |
| ACTION | 测试动作，例如发送、检查、设置、回复 |
| COMPONENT | 组件或部件 |
| EXPECTED_EXPR | 期望结果或响应表达 |
| FAULT_EXPR | 故障表达，例如非法值、超时、错误 ID |
| FIELD_NAME | 报文字段名 |
| INTERFACE | 充电接口或连接类型 |
| MESSAGE | 报文名称，例如 BMS、CRO、BEM、CRM |
| OBJECT | 测试对象，例如 BMS、充电机、车辆 |
| PARAM_NAME | 参数名称 |
| PARAM_VALUE | 参数取值 |
| SIGNAL | 信号名称，例如 CP、CC |
| TEST_CONDITION | 测试条件或阶段性前置条件 |

单标签分类任务包括：

| 分类任务 | 说明 |
| --- | --- |
| standard_source | 标准来源，例如 GB/T 34658-2025、GB/T 34657.2-2017 |
| scene_type | 交流 AC 或直流 DC 场景 |
| system_class | 系统类别 |
| test_layer | 测试层级，例如应用层、互操作 |
| condition_type | 正常或故障条件 |
| test_type | 正向或负向测试 |
| test_stage | 充电测试阶段 |
| fault_type | 故障类型 |

多标签分类任务为 `action_intent`，用于表达一句测试需求中可能同时出现的多个测试意图。

## 3. 模型结构

本次实验实现的模型位于：

- `src/ev_charge_testcase_generator/ml/model.py`
- `src/ev_charge_testcase_generator/ml/torch_dataset.py`
- `src/ev_charge_testcase_generator/ml/train.py`
- `src/ev_charge_testcase_generator/ml/predict.py`

模型主干采用 `hfl/chinese-macbert-base`，使用 `transformers.AutoModel` 加载。整体结构如下：

```text
自然语言测试需求
  -> AutoTokenizer 分词并保留 offset_mapping
  -> MacBERT 编码器
  -> GlobalPointer 实体抽取头
  -> 多个单标签分类头
  -> action_intent 多标签分类头
```

### 3.1 GlobalPointer 实体抽取头

实体抽取采用 GlobalPointer 思路，将实体识别转化为 token span 判断问题。对于每个实体类型，模型输出一个二维矩阵：

```text
[entity_type_count, max_length, max_length]
```

矩阵中第 `i, j` 个位置表示从第 `i` 个 token 到第 `j` 个 token 是否构成某类实体。相比传统 BIO 序列标注，GlobalPointer 能够直接建模实体起止边界，也便于表达不同实体类型的 span 打分。

训练数据中的实体标注是字符级 `start/end`，因此 `torch_dataset.py` 中使用 tokenizer 的 `return_offsets_mapping=True` 将字符区间映射为 token 区间，再构造 GlobalPointer 标签矩阵。

### 3.2 多任务分类头

句级分类任务使用 MacBERT 的 `[CLS]` 向量作为句子表示。每个单标签分类任务对应一个线性分类头，训练时使用交叉熵损失。`action_intent` 是多标签任务，对应一个线性分类头，训练时使用 `BCEWithLogitsLoss`。

### 3.3 损失函数

总损失由三部分组成：

```text
loss = entity_loss + classification_loss + action_intent_loss
```

其中：

- `entity_loss`：GlobalPointer 多标签 span 损失，本实验使用 BCEWithLogitsLoss，并增加实体损失权重与正样本权重。
- `classification_loss`：多个单标签分类任务的 CrossEntropyLoss 之和。
- `action_intent_loss`：多标签动作意图的 BCEWithLogitsLoss。

由于实体 span 矩阵中负样本远多于正样本，初始训练时模型容易倾向于不预测实体，导致实体 F1 为 0。因此正式训练加入：

- `entity_loss_weight = 5`
- `entity_positive_weight = 20`

该设置用于提高实体抽取任务和实体正样本在总损失中的影响。

## 4. 训练环境与配置

本次实验在本机 GPU 环境中完成。

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows |
| Python 环境 | conda env: `ev_charge_ml` |
| Python 版本 | 3.10 |
| GPU | NVIDIA GeForce GTX 1650 |
| PyTorch | 2.0.1+cu117 |
| Transformers | 4.40.2 |
| NumPy | 1.26.4 |
| 模型 | hfl/chinese-macbert-base |
| 最大长度 | 256 |
| batch size | 1 |
| epoch | 5 |
| learning rate | 2e-5 |
| device | cuda |

正式训练命令如下：

```powershell
conda run -n ev_charge_ml python -m ev_charge_testcase_generator.ml.train `
  --train data/semantic_dataset/train.jsonl `
  --dev data/semantic_dataset/dev.jsonl `
  --label-vocab data/semantic_dataset/label_vocab.json `
  --model-name hfl/chinese-macbert-base `
  --output-dir outputs/macbert_globalpointer_weighted `
  --epochs 5 `
  --batch-size 1 `
  --learning-rate 2e-5 `
  --max-length 256 `
  --device cuda `
  --entity-loss-weight 5 `
  --entity-positive-weight 20 `
  --entity-thresholds=-4,-3,-2,-1,0
```

模型输出目录：

```text
outputs/macbert_globalpointer_weighted
```

主要产物包括：

| 文件 | 说明 |
| --- | --- |
| best_model.pt | dev 集选择得到的最佳模型权重 |
| last_model.pt | 最后一个 epoch 的模型权重 |
| training_metrics.json | 每个 epoch 的训练损失和 dev 指标 |
| dev_metrics.json | dev 集最终评估结果 |
| test_metrics.json | test 集最终评估结果 |
| dev_error_analysis.json | 原始预测错误分析 |
| dev_error_analysis_postprocessed.json | 领域后处理后的错误分析 |
| label_vocab.json | 保存的标签词表 |
| tokenizer.json / vocab.txt | 保存的 tokenizer 文件 |

## 5. 训练过程观察

训练过程中，实体抽取指标随着 epoch 增加明显提升。以下为未加入领域后处理、仅使用模型输出和阈值选择时的 dev 集实体结果：

| Epoch | Train Loss | 阈值 | Precision | Recall | F1 | 预测实体数 | Gold 实体数 | TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.8640 | -2.0 | 0.0151 | 0.4757 | 0.0293 | 6477 | 206 | 98 |
| 2 | 3.7898 | 0.0 | 0.3833 | 0.5340 | 0.4462 | 287 | 206 | 110 |
| 3 | 2.6698 | 0.0 | 0.4704 | 0.6553 | 0.5477 | 287 | 206 | 135 |
| 4 | 1.9643 | 0.0 | 0.4696 | 0.7864 | 0.5880 | 345 | 206 | 162 |
| 5 | 1.4259 | 0.0 | 0.4818 | 0.8350 | 0.6110 | 357 | 206 | 172 |

可以看到，训练初期模型召回较高但误报非常多；随着训练推进，预测实体数量显著下降，precision 快速提高，recall 保持在较高水平。第 5 个 epoch 的未后处理实体 F1 达到 `0.6110`，说明实体抽取头已经学到较稳定的测试需求片段边界。

分类任务整体收敛较快。到第 5 个 epoch，dev 集大部分分类任务准确率达到 `1.0`，其中 `test_stage` 为 `0.8333`，`fault_type` 为 `0.9667`。这说明句级分类任务相对实体边界识别更容易，尤其在当前数据模板化程度较高的情况下，MacBERT 的句向量足以捕获主要类别差异。

## 6. 最终评估结果

最终评估使用 `outputs/macbert_globalpointer_weighted/best_model.pt`，并启用领域后处理逻辑。后处理主要用于约束报文名、动作词、对象词等高规律性实体，减少 GlobalPointer 在小数据场景下产生的过长 span。

### 6.1 Dev 集结果

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 30 |
| 实体 Precision | 0.9243 |
| 实体 Recall | 0.8301 |
| 实体 F1 | 0.8747 |
| 预测实体数 | 185 |
| Gold 实体数 | 206 |
| 实体 TP | 171 |
| action_intent Precision | 1.0000 |
| action_intent Recall | 1.0000 |
| action_intent Micro F1 | 1.0000 |

Dev 集单标签分类准确率：

| 分类任务 | Accuracy |
| --- | ---: |
| standard_source | 1.0000 |
| scene_type | 1.0000 |
| system_class | 1.0000 |
| test_layer | 1.0000 |
| condition_type | 1.0000 |
| test_type | 1.0000 |
| test_stage | 0.8333 |
| fault_type | 0.9667 |

### 6.2 Test 集结果

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 30 |
| 实体 Precision | 0.8883 |
| 实体 Recall | 0.8267 |
| 实体 F1 | 0.8564 |
| 预测实体数 | 188 |
| Gold 实体数 | 202 |
| 实体 TP | 167 |
| action_intent Precision | 0.9712 |
| action_intent Recall | 1.0000 |
| action_intent Micro F1 | 0.9854 |

Test 集单标签分类准确率：

| 分类任务 | Accuracy |
| --- | ---: |
| standard_source | 1.0000 |
| scene_type | 1.0000 |
| system_class | 1.0000 |
| test_layer | 1.0000 |
| condition_type | 1.0000 |
| test_type | 1.0000 |
| test_stage | 0.9000 |
| fault_type | 0.9667 |

Test 集结果与 dev 集接近，说明当前模型在同分布测试需求上的泛化表现较稳定。实体 F1 从 dev 的 `0.8747` 到 test 的 `0.8564` 有轻微下降，但整体仍保持在可用于原型验证的水平。

## 7. 消融观察

本次实验中可以得到一个清晰的阶段性对比：

| 设置 | 实体 F1 | 现象说明 |
| --- | ---: | --- |
| 原始训练与默认阈值 | 0.0000 | span 正样本极少，模型倾向于不预测实体 |
| 加实体损失权重和正样本权重 | 0.6110 | 模型开始稳定学习实体边界，但仍存在过长 span 和误报 |
| 加领域后处理 | 0.8564（test） | 明显减少长 span 误报，报文、动作、对象等规则性实体更稳定 |

该结果说明，单纯依赖小规模训练数据学习所有实体边界仍然困难；结合领域知识进行后处理，可以显著提升模型输出的可用性。对于本项目这样的工程原型，较合理的路线不是完全替换规则系统，而是采用“深度学习模型 + 领域规则约束”的混合式语义抽取方案。

## 8. 错误分析

Dev 集后处理后的错误分析结果如下：

| 错误类型 | 数量 |
| --- | ---: |
| False Positive | 14 |
| False Negative | 35 |
| Boundary Error | 2 |
| Type Error | 0 |

False Positive 主要集中在：

| 实体类型 | 数量 |
| --- | ---: |
| ACTION | 9 |
| MESSAGE | 2 |
| FAULT_EXPR | 1 |
| OBJECT | 1 |
| EXPECTED_EXPR | 1 |

False Negative 主要集中在：

| 实体类型 | 数量 |
| --- | ---: |
| FIELD_NAME | 10 |
| PARAM_VALUE | 5 |
| TEST_CONDITION | 4 |
| EXPECTED_EXPR | 4 |
| PARAM_NAME | 3 |
| ACTION | 3 |
| FAULT_EXPR | 2 |
| SIGNAL | 2 |
| MESSAGE | 1 |
| INTERFACE | 1 |

从错误分布看，模型对 `MESSAGE`、`OBJECT`、常见 `ACTION` 的识别较稳定，因为这些实体通常有较明显的词形模式或固定词表。例如报文名多为大写字母缩写，常见动作词也较集中。

剩余错误主要来自 `FIELD_NAME`、`PARAM_VALUE`、`TEST_CONDITION` 等实体。这些实体存在三个特点：

1. 边界变化较大：字段名和测试条件通常是中文短语，长度和搭配不固定。
2. 与上下文强相关：同一个片段在不同句式下可能是字段名、参数名或期望表达的一部分。
3. 标注样本较少：部分细分类实体在训练集中出现频次低，模型难以学习稳定边界。

因此，后续如果继续优化，应该优先补强字段名和参数值相关知识，而不是继续盲目增加 epoch。

## 9. 结论

本次实验表明，MacBERT-GlobalPointer 多任务模型可以有效完成新能源汽车充电测试需求的语义抽取任务。分类任务表现稳定，实体抽取在加入损失权重和领域后处理后取得较高 F1。

主要结论如下：

1. 句级分类任务难度较低，MacBERT 编码器能够较好地区分标准来源、AC/DC 场景、条件类型、测试类型等类别。
2. 实体抽取任务难度明显高于分类任务，主要原因是 GlobalPointer span 矩阵中正负样本极度不均衡。
3. 实体损失权重和正样本权重能够显著缓解“不预测实体”的问题，使实体 F1 从接近 0 提升到约 0.61。
4. 领域后处理对测试需求抽取非常有效，尤其适合报文名、动作词、对象词等规则性较强的实体。
5. 当前系统更适合采用混合式方案：模型负责语义泛化，规则和知识库负责边界修正与合法性约束。

从论文实验角度看，该结果可以支撑如下观点：

> 面向新能源汽车充电测试需求的结构化生成任务，基于预训练语言模型的多任务语义抽取方法能够有效提升自然语言需求到结构化语义表示的自动化程度；同时，引入充电报文字段知识和领域后处理机制，可以进一步提升实体边界识别的精度和工程可用性。

## 10. 后续工作建议

后续建议优先做三件事：

1. 增加字段名词典约束：利用 `data/message_field_knowledge/dc_message_fields.json` 中的报文字段知识，对 `FIELD_NAME` 进行候选召回和边界修正。
2. 补充少样本实体数据：针对 `FIELD_NAME`、`PARAM_VALUE`、`TEST_CONDITION` 增加更多句式变体，提升模型对短实体和复杂中文短语的鲁棒性。
3. 增加消融实验脚本：保留 `无后处理`、`有后处理`、`不同实体权重` 的可重复评估入口，便于论文中展示方法组件贡献。

如果将模型接入规则版主流程，建议先以辅助模式接入：模型输出候选实体和分类结果，规则系统继续负责模板匹配、约束表达生成和最终测试用例校验。这样可以降低模型误判对最终测试用例质量的影响，也更符合当前原型系统的渐进式演进路线。
