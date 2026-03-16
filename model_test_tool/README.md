# 模型测试工具使用说明

## 首次使用配置

### 方法1：自动生成（推荐）⭐

1. 运行程序：
   ```bash
   python src/main.py
   ```
   或
   ```bash
   python run.py
   ```

2. 通过GUI配置：
   - 在"数据集"标签页添加你的JSONL数据集文件
   - 在"模型"标签点击"添加模型"并配置
   - 在"通用设置"标签调整评测参数

3. 配置会自动保存到`src/config.json`（已被.gitignore保护）

### 方法2：手动配置

1. 复制配置模板：
   ```bash
   cp config.example.json src/config.json
   ```

2. 编辑`src/config.json`，修改以下配置：
   ```json
   {
     "datasets": ["./data/your_dataset.jsonl"],  // 修改为你的数据集路径
     "output_dir": "./result",                    // 输出目录
     "sample_count": 100,                         // 每个模型评测样本数
     "models": ["model1", "model2"]               // 添加模型名称
   }
   ```

3. 配置模型参数（在GUI中或手动编辑model_configs部分）

## 配置文件说明

- **config.example.json**: 配置模板（提交到Git）
- **src/config.json**: 实际配置文件（自动生成，不提交）
- **.gitignore**: 已配置忽略src/config.json

## 模型配置指南

### 1. 添加模型
1. 运行程序后，在"模型"标签页点击"添加模型"按钮
2. 输入模型名称（例如：gpt-3.5-turbo、local-model等）
3. 系统会自动为新模型创建默认配置

### 2. 配置模型参数
1. 在"模型"标签页中选择要配置的模型
2. 点击"配置模型"按钮打开配置对话框
3. 设置以下参数：
   - **API Base**: API基础地址（默认为OpenAI地址）
   - **API Key**: API密钥（可为空，如果使用本地模型）
   - **请求模型名称**: 实际请求的模型名称
   - **System Message**: 系统消息（可选）
   - **Temperature**: 温度参数（0.0-2.0）
   - **Max Tokens**: 最大令牌数
   - **Top P**: Top P参数（0.0-1.0）
   - **Frequency Penalty**: 频率惩罚（-2.0-2.0）
   - **Presence Penalty**: 存在惩罚（-2.0-2.0）

### 3. 常见配置示例

#### OpenAI GPT模型配置：
- API Base: https://api.openai.com/v1
- API Key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
- 请求模型名称: gpt-3.5-turbo

#### 本地模型配置（如LM Studio）：
- API Base: http://localhost:1234/v1
- API Key: （可以留空）
- 请求模型名称: （根据实际模型名称填写）

#### Claude模型配置：
- API Base: https://api.anthropic.com/v1
- API Key: sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
- 请求模型名称: claude-3-haiku-20240307

### 4. 故障排除

#### 503错误（服务不可用）：
- 检查API服务是否正常运行
- 确认API Base地址是否正确
- 检查网络连接是否正常
- 如果是本地模型，确认本地服务是否已启动

#### 配置相关问题：
1. 检查config.json文件中的model_configs部分是否包含模型配置
2. 确保API Base地址正确
3. 确保API Key有效（如果需要）
4. 检查请求模型名称是否正确

### 5. 保存配置
配置会自动保存，也可以手动点击"保存配置"按钮。

## 运行评测
1. 在"评测"标签页选择要测试的数据集和模型
2. 点击"开始评测"按钮
3. 查看评测结果和生成的报告文件

## 结果文件说明
评测完成后会生成三个结果文件：
1. **model_scores_*.txt**: 包含各模型的总分
2. **test_results_archive_*.txt**: 包含详细的评测结果摘要
3. **complete_test_results_*.json**: 包含完整的原始数据，包括：
   - 原始题目内容
   - 各个模型的回答答案
   - 正确答案
   - 每次评测的得分
   - 最终汇总分数

## 注意事项
- 确保数据集文件格式正确（JSONL格式）
- 输出目录需要有写入权限
- 评测过程中可以点击"停止评测"按钮中断评测
- 评测结果会保存到指定的输出目录中
- config.json包含你的API密钥，不要提交到Git仓库
