# Tool Calling

`get_sft_data`、`inference`、`get_dpo_data` 围绕函数调用（tool calling）数据构建与训练展开：

- `get_sft_data`：把原始函数调用数据转成 SFT 训练格式。
- `inference`：把基准集改造成推理输入，调用模型推理并产出预测结果。
- `get_dpo_data`：将推理结果与标准答案对比，生成 DPO 训练偏好数据。

可以理解为两条主线：

1. SFT 数据线：原始数据 -> ShareGPT 转换 -> CoT 增强。
<!-- 1. SFT 数据线：原始数据 -> ShareGPT 转换 -> CoT 增强 -> 真实 API 执行补 observation。 -->
2. DPO 数据线：基准集 -> 推理输入/标准答案 -> 模型推理 -> 结果对比 -> DPO 偏好样本。

注：文件结构发生过调整，实际调用时需要修改文件路径

## 数据和模型来源

- sft: APIGen的xlam_function_calling_60k.json，GPT的alpaca_gpt4_data.json
- dpo: 智谱的ComplexFuncBench.jsonl
- 基础模型: Qwen3-4b-think-2507

## get_sft_data

### `xlam2shareGPT.py`
- 将原始 `xlam_function_calling_60k.json` 转换为训练用 ShareGPT 风格结构，把 `query/answers/tools` 转成 `conversations + tools + category`。
- 负责工具参数 schema 归一化（类型映射、required 处理）。

### `addCoT.py`
在转换后的样本上补充 CoT（`<think>...</think>`），产出带思维链的 SFT 数据。

<!-- ### `run_rapidapi_calls.py`
扫描数据集中的函数调用，自动生成 RapidAPI 注册模板（函数到 HTTP 请求配置映射），并将执行结果写回 `observation` / `execution_results`。 -->

## inference

### `get_question.py`
从 `ComplexFuncBench.jsonl` 重建推理输入集与标准答案。
- `inference_question.json`：给模型推理的 messages + tools。
- `possible_answer.json`：用于评测的 ground truth。
- 同时把函数定义转换为 OpenAI tool 调用所需结构。

### `inference.py`
调用模型接口进行批量推理，解析输出中的 `<tool_call>...</tool_call>`，规范化为 `tool_calls` 列表，并按 JSONL 写出 `inference_answer.json`，并保留原始响应与可选 reasoning。

## get_dpo_data

### `compare.py`
- 对齐并比较 `possible_answer.json` 与 `inference_answer.json`。
- 统一格式后判断预测是否正确，输出差异细节。
- 处理缺失推理条目，补全为 `complete_inference_answer.json`，最终产出 `comparison.json`。

### `prompt.py`
- 定义 DPO 数据构造与判别用的提示词模板。
- 包含正例 CoT、负例生成、工具调用判别、文本总结判别等模板。

### `client.py`
- 封装“教师模型”调用函数：
  - 生成 DPO 正负样本片段；
  - 判断预测与真值一致性；
  - 生成拒绝样本与 CoT。
- 是 `get_cot.py` 的底层 API 调用模块。

### `get_cot.py`
- DPO 数据主构建脚本。
- 读取原始上下文 + 比对结果，根据正确/错误情况调用教师模型：
  - 生成 `chosen`（正例）；
  - 构造或生成 `rejected`（负例）。
- 产出 `dpo_training.json`。

### `dpo2sharegpt.py`
- 将 `dpo_training.json` 再转换为训练侧更易消费的格式。
- 对 `chosen/rejected` 字段做结构清洗，统一为字符串内容（含 `<think>` 包裹）。


## 执行顺序

### SFT 线
1. `get_sft_data/xlam2shareGPT.py`
2. `get_sft_data/addCoT.py`
<!-- 3. `get_sft_data/run_rapidapi_calls.py` -->

### DPO 线
1. `inference/get_question.py`
2. `inference/inference.py`
3. `get_dpo_data/compare.py`
4. `get_dpo_data/get_cot.py`
5. `get_dpo_data/dpo2sharegpt.py`