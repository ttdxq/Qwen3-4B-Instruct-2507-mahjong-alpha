# Qwen3-4B-Instruct-2507-mahjong-alpha

[English](./README_en.md)

[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-models-green)](https://huggingface.co/TTDXQ/Qwen3-4B-Instruct-2507-mahjong-alpha) [![Model Size](https://img.shields.io/badge/Model%20Size-4B-blue)](https://huggingface.co/TTDXQ/Qwen3-4B-Instruct-2507-mahjong-alpha)

`Qwen3-4B-Instruct-2507-mahjong-alpha` 是一个基于 `unsloth/Qwen3-4B-Instruct-2507` 进行 QLoRA 微调的立直麻将垂直模型，面向四麻弃牌建议任务。

模型可根据输入的场次信息、手牌、副露、牌河、牌效与防守信息，输出当前最应打出的一张牌。

当前版本主要面向工具集成场景，推理输出为单张牌文本，不包含解释信息。

## 项目状态

- 任务：四麻立直麻将弃牌建议
- 基座模型：`unsloth/Qwen3-4B-Instruct-2507`
- 微调方式：`QLoRA`
- 训练框架：`Unsloth`
- 发布格式：`GGUF (F16)`
- 推理方式：`llama.cpp`
- 维护者：`TTDXQ`

## 适用范围

本模型面向四麻场景，不含赤宝牌。当前版本专注于“弃牌建议”这一单一任务，不提供完整对局规划，也不提供役种、打点或详细攻防解释。

## 使用限制

- 仅支持弃牌建议
- 不支持完整对局规划
- 不支持役种、打点、进攻防守解释
- 不保证竞赛或实战效果
- 仅供研究与学习使用

## 禁止用途

禁止将本模型用于：

- 作弊
- 外挂
- 代打
- 真钱赌博辅助

## 模型输入输出

### 输入格式

模型输入为结构化自然语言局面描述。示例：

```text
[情景分析]
- 牌局: 东一局，你是庄家 (第1巡，牌墙余69张)。
- 状态: 当前排名 1/4 (与一位差 0)。
- 宝牌: 5万
- 各玩家分数: 你有 25分, 下家: 25分, 对家: 25分, 上家: 25分。
- 你的手牌: 1万 5万 7万 3筒 5筒 6筒 8筒 8筒 3索 5索 8索 南 白 发
- 牌效: 5 向听，进张 82 张。
- 防御：
  最安全牌放铳率：11.3%
  平均放铳率：18.5%
  最危险牌放铳率：25.9%
场上已见牌信息
各玩家副露信息:本家副露：无, 下家副露：无, 对家副露：无, 上家副露：无
各玩家牌河信息:本家：无, 下家：无, 对家：无, 上家：无

[任务]
根据当前情景，选择一张最应该打出的手牌。
```

### 输出格式

模型输出严格为“单张牌文本”，不带“打”字，不带解释。例如：

```text
白
```

## 数据集

训练数据使用 `pjura/mahjong_board_states` 的 2018 年部分数据。该数据集来源于天风麻将的游玩记录，每条数据包含 511 个数据点，涵盖游戏基础信息、宝牌指示牌、视角玩家手牌、玩家副露、牌河信息、玩家舍牌、弃牌决策等。

### 数据处理

将原始数据转换为便于阅读的自然语言描述形式，并根据数据计算出巡目数、实际宝牌、简易放铳参考等信息。根据巡目调整样本比例：

- 1~3 巡：15%
- 4~6 巡：20%
- 7~12 巡：35%

最终使用 `192000` 条样本，未混入通用指令数据或自建数据。

- 训练集：`192000`
- 验证集：`2000`
- 测试集：`2019 年数据按需抽取`
- 训练 / 验证 / 测试：完全互不重叠

### 数据集引用

```bibtex
@dataset{mahjong_board_states,
  title = {MahJong Board States Dataset},
  author = {Patrick Jura},
  year = {2024},
  url = {https://huggingface.co/datasets/pjura/mahjong_board_states}
}
```

## 训练信息

### 模型配置
- 基础模型：`unsloth/Qwen3-4B-Instruct-2507`
- 训练加载精度：`4bit`
- 微调方式：`QLoRA`
- 训练框架：`Unsloth`
- Max sequence length：`2048`

### LoRA 参数
- Rank：`128`
- Alpha：`256`
- 目标模块：全部

### 训练超参数
- Learning rate：`1e-4`
- LR scheduler：`cosine`
- Batch size：`64`
- 单卡批次：`2`
- 梯度累积步数：`32`
- Training steps：`3000`
- Warmup steps：`300`
- Random seed：`3407`
- 加载最优检查点：是

### 训练时间
- 总时长：约 16.44 小时

## 评测设计

当前评测分为两类：

1. 与数据库中的弃牌动作进行对比
2. 与 `Mortal` 推荐结果进行一致率对比

其中，数据库动作评测分为三个场景：

### 1. 牌效测试

- 早巡：60%
- 中巡：30%
- 晚巡：10%

主要考验模型构建手牌的能力。

### 2. 防守测试

- 早巡：10%
- 中巡：50%
- 晚巡：40%

主要考验模型寻找安全牌、判断攻守与弃和的能力。

### 3. 综合测试

- 早巡：30%
- 中巡：45%
- 晚巡：25%

综合模拟对局，考验模型综合能力。

巡目定义：

- 1~6 巡：早巡
- 7~12 巡：中巡
- 13 巡及以后：晚巡

## 评测结果

### 与数据库弃牌动作对比

推理参数：Temperature=0.1, Top_P=0.1

#### 牌效测试

| 模型 | 方法 | 得分 | 样本全对率 | 样本零分率 |
|------|------|------|------------|------------|
| Qwen3-4B | 提示词工程 | 50.21 | 6.60% | 86.13% |
| Qwen3-4B | 微调 | 229.66 | 45.87% | 53.93% |
| DeepSeek-V3.2 | 提示词工程 | 181.66 | 21.40% | 46.33% |

#### 防守测试

| 模型 | 方法 | 得分 | 样本全对率 | 样本零分率 |
|------|------|------|------------|------------|
| Qwen3-4B | 提示词工程 | 53.55 | 6.17% | 84.43% |
| Qwen3-4B | 微调 | 239.89 | 47.93% | 52.00% |
| DeepSeek-V3.2 | 提示词工程 | 172.00 | 16.00% | 46.80% |

#### 综合测试

| 模型 | 方法 | 得分 | 样本全对率 | 样本零分率 |
|------|------|------|------------|------------|
| Qwen3-4B | 提示词工程 | 53.44 | 0.60% | 84.40% |
| Qwen3-4B | 微调 | 233.33 | 46.53% | 53.20% |
| DeepSeek-V3.2 | 提示词工程 | 179.44 | 18.07% | 44.93% |

### 与 Mortal 对比

推理参数：Temperature=0.6, Top_P=0.95

#### 测试1：全部巡目数据

- 样本数：3000
- Top-1 准确率：**50.73%**
- Top-3 准确率：**83.37%**

#### 测试2：去除早巡数据

- 有效样本数：3000
- Top-1 准确率：**48.70%**
- Top-3 准确率：**79.20%**

> 注：Mortal 是当前开源最强的立直麻将 AI 之一

## 推理方式

### llama.cpp / llama-server

```bash
.\llama-server -m Qwen3-4B-Instruct-2507-mahjong-alpha.gguf -c 2048
```

## 工具使用

本项目提供以下工具，用于数据处理和模型评测：

### 1. 模型测试工具 (model_test_tool/)

基于 PySide6 的图形化模型评测工具，支持多模型并发测试和结果分析。

**主要功能：**
- 📊 可视化配置数据集和模型
- ⚡ 支持多模型并发评测
- 📈 实时显示评测进度和结果
- 💾 自动保存评测结果
- 🔧 灵活的评分和过滤配置

**快速开始：**
```bash
cd model_test_tool
pip install -r requirements.txt
python src/main.py
```

**详细文档：** 请查看 [model_test_tool/README.md](./model_test_tool/README.md)

**支持的场景：**
- OpenAI API 兼容接口
- 本地模型服务（如 llama.cpp、LM Studio）
- 自定义 API Base 和参数配置

---

### 2. 数据转换工具 (process_parquet.py)

将 Parquet 格式的麻将数据转换为 JSONL 格式，并自动计算牌效指标。

**主要功能：**
- 🔄 Parquet → JSONL 格式转换
- 🎯 自动计算向听数（Shanten）
- 🔢 自动计算进张数（Ukeire）
- 🛡️ 计算安全牌/危险牌评分
- ⚡ 支持多进程并行处理

**使用方法：**
```bash
# 处理单个文件
python process_parquet.py input.parquet output.jsonl

# 处理整个文件夹
python process_parquet.py input_folder/ output_folder/

# 限制处理数量
python process_parquet.py input.parquet output.jsonl --max=10000
```

**输出格式：**
```json
{
  "text": "[情景分析]\n- 牌局: 东一局...\n[任务]\n根据当前情景，选择一张最应该打出的手牌。\n白"
}
```

---

### 3. 数据分割工具 (random_split_jsonl.py)

将大型 JSONL 文件随机分割为多个固定大小的文件。

**主要功能：**
- 🎲 随机打乱数据顺序
- ✂️ 按指定行数分割
- 💾 内存高效的索引机制
- ⚡ 支持多线程写入

**使用方法：**
```bash
python random_split_jsonl.py input.jsonl \
    --output_dir output_folder \
    --lines_per_file 10000 \
    --seed 42
```

**参数说明：**
- `--output_dir`: 输出目录（默认：output_jsonl）
- `--lines_per_file`: 每个文件的行数（必需）
- `--seed`: 随机种子（可选，用于可复现）
- `--workers`: 并发线程数（默认：1）

**输出示例：**
```
output_folder/
├── input_00000.jsonl
├── input_00001.jsonl
├── input_00002.jsonl
└── ...
```

---

### 4. 数据重组工具 (shuffle_and_split_jsonl.py)

按巡目比例重新平衡麻将数据集，确保各巡目数据分布合理。

**主要功能：**
- 📊 按巡目自动分类（序盘/构筑/对攻/尾巡）
- ⚖️ 智能比例重组（可配置）
- 🔄 全局洗牌打乱顺序
- ⚡ 多进程并行处理

**巡目分类：**
- 序盘（1-3巡）：15%
- 构筑（4-6巡）：20%
- 对攻（7-12巡）：30%
- 尾巡（13+巡）：35%

**使用方法：**
```bash
# 基本用法：自动重组数据集
python shuffle_and_split_jsonl.py input_data/ \
    --output_dir balanced_data/ \
    --lines_per_file 10000

# 仅分桶不重组
python shuffle_and_split_jsonl.py input_data/ \
    --output_dir split_data/ \
    --split_only
```

**参数说明：**
- `--output_dir`: 输出目录（默认：dataset_balanced）
- `--lines_per_file`: 每个文件的行数（默认：10000）
- `--max_files`: 最大输出文件数（默认：0=自动计算）
- `--workers`: 并发进程数（默认：CPU核心数）
- `--split_only`: 仅分桶输出，不进行比例重组

**输出结构：**
```
balanced_data/
├── train_balanced_00000.jsonl  # 重组后的平衡数据
├── train_balanced_00001.jsonl
└── ...
```

---

## 仓库链接

- GitHub：https://github.com/ttdxq/Qwen3-4B-Instruct-2507-mahjong-alpha
- Hugging Face：https://huggingface.co/TTDXQ/Qwen3-4B-Instruct-2507-mahjong-alpha


## License

本仓库遵循基座模型所使用的许可证（Apache License 2.0）。

训练数据来自 `pjura/mahjong_board_states`，其许可证为 `CC BY 4.0`，使用时请保留相应署名与引用。

## Acknowledgements

感谢以下开源资源：

- `unsloth/Qwen3-4B-Instruct-2507`
- `pjura/mahjong_board_states`
- `Mortal`
