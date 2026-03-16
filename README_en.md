# Qwen3-4B-Instruct-2507-mahjong-alpha

[中文](./README.md)

[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-models-green)](https://huggingface.co/TTDXQ/Qwen3-4B-Instruct-2507-mahjong-alpha) [![Model Size](https://img.shields.io/badge/Model%20Size-4B-blue)](https://huggingface.co/TTDXQ/Qwen3-4B-Instruct-2507-mahjong-alpha)

`Qwen3-4B-Instruct-2507-mahjong-alpha` is a Riichi Mahjong domain model fine-tuned from `unsloth/Qwen3-4B-Instruct-2507` with QLoRA.

It is designed for 4-player Riichi Mahjong discard recommendation: given round information, hand tiles, calls, visible tiles, tile-efficiency, and defense signals, the model outputs the single best discard tile for the current state.

The current version is mainly intended for tool integration. The output is a single tile text only, without explanation.

## Project Status

- Task: 4-player Riichi Mahjong discard recommendation
- Base model: `unsloth/Qwen3-4B-Instruct-2507`
- Fine-tuning: `QLoRA`
- Training framework: `Unsloth`
- Release format: `GGUF (F16)`
- Inference: `llama.cpp`
- Maintainer: `TTDXQ`

## Scope

This model targets 4-player Riichi Mahjong without red dora. The current version focuses only on discard recommendation. It does not provide full-game planning, yaku/score analysis, or detailed offense-defense explanations.

## Limitations

- Discard recommendation only
- No full-game planning
- No yaku, point calculation, or detailed strategic explanation
- Not guaranteed for competitive or real-match performance
- For research and learning purposes only

## Prohibited Uses

This model must not be used for:

- cheating
- game automation or plug-ins
- account boosting or ghost-playing
- real-money gambling assistance

## Input and Output

### Input Format

The model input is a structured natural-language game-state description. Example:

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

### Output Format

The output is strictly a single tile text without any prefix like "discard" and without explanation. Example:

```text
白
```

## Dataset

The training data uses the 2018 subset of `pjura/mahjong_board_states`. This dataset originates from Tenhou.net gameplay records, with each record containing 511 data points covering game basics, dora indicators, player hand tiles, calls, discard piles, and discard decisions.

### Data Processing

The raw data was converted into human-readable natural language descriptions, with calculated turn numbers, actual dora, and simplified risk assessment. Sample distribution by turn:

- Turns 1-3: 15%
- Turns 4-6: 20%
- Turns 7-12: 35%

A total of `192000` samples were used, with no general instruction data or self-built data mixed in.

- Train: `192000`
- Validation: `2000`
- Test: sampled as needed from 2019 data
- Train / validation / test are fully non-overlapping

### Dataset Citation

```bibtex
@dataset{mahjong_board_states,
  title = {MahJong Board States Dataset},
  author = {Patrick Jura},
  year = {2024},
  url = {https://huggingface.co/datasets/pjura/mahjong_board_states}
}
```

## Training Details

### Model Configuration
- Base Model: `unsloth/Qwen3-4B-Instruct-2507`
- Training Precision: `4bit`
- Fine-tuning Method: `QLoRA`
- Framework: `Unsloth`
- Max Sequence Length: `2048`

### LoRA Parameters
- Rank: `128`
- Alpha: `256`
- Target Modules: All

### Training Hyperparameters
- Learning Rate: `1e-4`
- LR Scheduler: `cosine`
- Batch Size: `64`
- Per-device Batch: `2`
- Gradient Accumulation Steps: `32`
- Training Steps: `3000`
- Warmup Steps: `300`
- Random Seed: `3407`
- Load Best Checkpoint: Yes

## Evaluation Protocol

Current evaluation has two parts:

1. Comparison against discard actions from the dataset
2. Agreement rate against `Mortal` recommendations

The dataset-action evaluation is divided into three scenarios:

### 1. Tile-Efficiency Test

- Early: 60%
- Mid: 30%
- Late: 10%

This test focuses on hand-building ability.

### 2. Defense Test

- Early: 10%
- Mid: 50%
- Late: 40%

This test focuses on safe-tile search and offense-defense judgment.

### 3. Comprehensive Test

- Early: 30%
- Mid: 45%
- Late: 25%

This test simulates more complete game situations.

Turn-stage definition:

- Turn 1-6: early
- Turn 7-12: mid
- Turn 13 and later: late

## Results

### Comparison with Dataset Actions

Inference parameters: Temperature=0.1, Top_P=0.1

Metrics explanation:
- Score: Max 500 points (1 point per correct sample, 0 for incorrect)
- Full-match rate: Samples where all 3 tests matched the dataset
- Zero-score rate: Samples where all 3 tests disagreed with the dataset

#### Tile-Efficiency Test

| Model | Method | Score | Full-match Rate | Zero-score Rate |
|-------|--------|-------|----------------|-----------------|
| Qwen3-4B | Prompt Engineering | 50.21 | 6.60% | 86.13% |
| Qwen3-4B | Fine-tuned | 229.66 | 45.87% | 53.93% |
| DeepSeek-V3.2 | Prompt Engineering | 181.66 | 21.40% | 46.33% |

#### Defense Test

| Model | Method | Score | Full-match Rate | Zero-score Rate |
|-------|--------|-------|----------------|-----------------|
| Qwen3-4B | Prompt Engineering | 53.55 | 6.17% | 84.43% |
| Qwen3-4B | Fine-tuned | 239.89 | 47.93% | 52.00% |
| DeepSeek-V3.2 | Prompt Engineering | 172.00 | 16.00% | 46.80% |

#### Comprehensive Test

| Model | Method | Score | Full-match Rate | Zero-score Rate |
|-------|--------|-------|----------------|-----------------|
| Qwen3-4B | Prompt Engineering | 53.44 | 0.60% | 84.40% |
| Qwen3-4B | Fine-tuned | 233.33 | 46.53% | 53.20% |
| DeepSeek-V3.2 | Prompt Engineering | 179.44 | 18.07% | 44.93% |

### Comparison with Mortal

Inference parameters: Temperature=0.6, Top_P=0.95

#### Test 1: All Turn Data

- Samples: 3000
- Top-1 Accuracy: **50.73%**
- Top-3 Accuracy: **83.37%**

#### Test 2: Excluding Early Turns

- Valid Samples: 3000
- Top-1 Accuracy: **48.70%**
- Top-3 Accuracy: **79.20%**

> Note: Mortal is one of the strongest open-source Riichi Mahjong AIs currently available

## Inference

### llama.cpp / llama-server

```bash
.\llama-server -m Qwen3-4B-Instruct-2507-mahjong-alpha.gguf -c 2048
```

## Tools

This repository includes the following tools for data processing and model evaluation:

### 1. Model Testing Tool (model_test_tool/)

A GUI-based model evaluation tool powered by PySide6, supporting multi-model concurrent testing and result analysis.

**Key Features:**
- 📊 Visual configuration for datasets and models
- ⚡ Multi-model concurrent evaluation
- 📈 Real-time progress and result display
- 💾 Automatic result saving
- 🔧 Flexible scoring and filtering options

**Quick Start:**
```bash
cd model_test_tool
pip install -r requirements.txt
python src/main.py
```

**Documentation:** See [model_test_tool/README.md](./model_test_tool/README.md)

**Supported Scenarios:**
- OpenAI API-compatible endpoints
- Local model servers (e.g., llama.cpp, LM Studio)
- Custom API Base and parameter configuration

---

### 2. Data Conversion Tool (process_parquet.py)

Converts Mahjong data from Parquet format to JSONL format, with automatic tile-efficiency calculations.

**Key Features:**
- 🔄 Parquet → JSONL format conversion
- 🎯 Automatic Shanten (tile efficiency) calculation
- 🔢 Automatic Ukeire (effective tiles) calculation
- 🛡️ Safe/dangerous tile scoring
- ⚡ Multi-process parallel processing

**Usage:**
```bash
# Process single file
python process_parquet.py input.parquet output.jsonl

# Process entire folder
python process_parquet.py input_folder/ output_folder/

# Limit processing count
python process_parquet.py input.parquet output.jsonl --max=10000
```

**Output Format:**
```json
{
  "text": "[情景分析]\n- 牌局: 东一局...\n[任务]\n根据当前情景，选择一张最应该打出的手牌。\n白"
}
```

---

### 3. Data Splitting Tool (random_split_jsonl.py)

Randomly splits large JSONL files into multiple fixed-size files.

**Key Features:**
- 🎲 Random shuffling of data
- ✂️ Split by specified line count
- 💾 Memory-efficient indexing
- ⚡ Multi-threaded writing support

**Usage:**
```bash
python random_split_jsonl.py input.jsonl \
    --output_dir output_folder \
    --lines_per_file 10000 \
    --seed 42
```

**Parameters:**
- `--output_dir`: Output directory (default: output_jsonl)
- `--lines_per_file`: Lines per file (required)
- `--seed`: Random seed (optional, for reproducibility)
- `--workers`: Concurrent threads (default: 1)

**Output Example:**
```
output_folder/
├── input_00000.jsonl
├── input_00001.jsonl
├── input_00002.jsonl
└── ...
```

---

### 4. Data Rebalancing Tool (shuffle_and_split_jsonl.py)

Rebalances Mahjong dataset by turn distribution to ensure reasonable data distribution across game stages.

**Key Features:**
- 📊 Automatic turn-based categorization (early/building/attack/late)
- ⚖️ Intelligent proportional rebalancing (configurable)
- 🔄 Global shuffling
- ⚡ Multi-process parallel processing

**Turn Categorization:**
- Early (turns 1-3): 15%
- Building (turns 4-6): 20%
- Attack (turns 7-12): 30%
- Late (turns 13+): 35%

**Usage:**
```bash
# Basic: automatic rebalancing
python shuffle_and_split_jsonl.py input_data/ \
    --output_dir balanced_data/ \
    --lines_per_file 10000

# Categorization only, no rebalancing
python shuffle_and_split_jsonl.py input_data/ \
    --output_dir split_data/ \
    --split_only
```

**Parameters:**
- `--output_dir`: Output directory (default: dataset_balanced)
- `--lines_per_file`: Lines per file (default: 10000)
- `--max_files`: Max output files (default: 0=auto-calculate)
- `--workers`: Concurrent processes (default: CPU cores)
- `--split_only`: Categorize only, no proportional rebalancing

**Output Structure:**
```
balanced_data/
├── train_balanced_00000.jsonl  # Rebalanced data
├── train_balanced_00001.jsonl
└── ...
```

---

## Repository Links

- GitHub: https://github.com/ttdxq/Qwen3-4B-Instruct-2507-mahjong-alpha
- Hugging Face: https://huggingface.co/TTDXQ/Qwen3-4B-Instruct-2507-mahjong-alpha


## License

This repository follows the license used by the base model (Apache License 2.0).

The training data comes from `pjura/mahjong_board_states`, which is licensed under `CC BY 4.0`. Please preserve the required attribution and citation when using it.

## Acknowledgements

Thanks to the following open-source resources:

- `unsloth/Qwen3-4B-Instruct-2507`
- `pjura/mahjong_board_states`
- `Mortal`
