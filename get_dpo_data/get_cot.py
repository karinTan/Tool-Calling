import json
import random
from typing import List, Dict
import openai
from openai import OpenAI
from tqdm import tqdm
import re
from prompt import ERROR_TYPES, POSITIVE_COT_TOOL, POSITIVE_COT_STR, JUDGE_PROMPT_STR, NEGATIVE_STR, JUDGE_PROMPT_TOOL, NEGATIVE_TOOL, GENERATE_COT_TOOL, GENERATE_COT_STR, POSITIVE_COT_TOOL_OUTPUT, NEGATIVE_TOOL_OUTPUT
from client import generate_dpo_str, generate_dpo_tool, judge_tool, generate_rejected_tool, judge_str, generate_rejected_str, generate_cot

# ================= 配置区域 =================
TEACHER_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEACHER_API_KEY = "sk-336804f6c9144365b4783cf362449182" 
MODEL_NAME = "qwen-plus-2025-07-14" 

# 文件路径
RAW_DATA_PATH = "/Users/tankling/all_my_files/coding/qwen/inference/inference_question.json"    # 原始上下文
COMPARISON_DATA_PATH = "/Users/tankling/all_my_files/coding/qwen/inference/comparison.json"      # 比对结果（包含 flag, answer, ground_truth）
OUTPUT_PATH = "/Users/tankling/all_my_files/coding/qwen/inference/dpo_training.json"     # 最终输出

def get_json(answer: str):
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, answer, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            # 尝试解析JSON
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print("JSON解析错误:", e)
            return None

def train_format(answer: List[Dict]):
    train_answer = []
    for func in answer:
        func_name = func["function"]["name"]
        args_str = func["function"]["arguments"]
        args_dict = json.loads(args_str)
        train_item = {
            "name": func_name,
            "arguments": args_dict
        }
        train_answer.append(train_item)
    return train_answer

def train_tools_format(tools: str):
    tools_list = json.loads(tools)
    train_tools = []
    for tool in tools_list:
        train_tools.append(tool["function"])
    return train_tools

def get_error(type):
    if type=="list":
        error_type = random.choice(["Nested function calls", "Short dependency", "Long dependency", "Missed function or parameters"])
        error_def = ERROR_TYPES[error_type]
    else:
        error_type = random.choice(["Wrong summarization", "Partially relevant"])
        error_def = ERROR_TYPES[error_type]
    return error_type, error_def

# ================= 主流程 =================

def build_dataset():
    client = OpenAI(base_url=TEACHER_API_BASE, api_key=TEACHER_API_KEY)
    
    # 加载数据
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = {item['id']: item for item in json.load(f)} # 转为字典方便索引
    
    comparison_list: List[Dict] = []
    with open(COMPARISON_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            comparison_item = json.loads(line)
            comparison_list.append(comparison_item)

    final_dataset = []

    print(f"开始处理 {len(comparison_list)} 条数据...")

    last_id = -1
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            train_data = json.loads(line)
            last_id = train_data["id"] if last_id < train_data["id"] else last_id

    mode = 'a' if last_id > -1 else 'w'
    with open(OUTPUT_PATH, mode, encoding='utf-8') as f:
        for item in tqdm(comparison_list):
            # comparison信息
            data_id: int = item['id']
            if data_id <= last_id:
                continue
            flag: bool = item['flag']
            ground_truth = item['ground_truth']
            answer = item['answer']
            reasoning: str = item.get('reasoning', "") # 模型原本的 CoT

            # question信息
            raw_entry = raw_data.get(data_id)
            if not raw_entry:
                continue
            messages: List[Dict] = raw_entry["messages"]
            messages_str = json.dumps(messages, ensure_ascii=False)
            tools: str = raw_entry["tools"]
            train_tools = json.dumps(train_tools_format(tools), ensure_ascii=False)

            # 基础结构
            system_prompt: str = ""
            conversations: List[Dict] = []
            for message in messages:
                role = message["role"]
                if role=="system":
                    value: str = message["content"]
                    system_prompt = value
                elif role=="user":
                    value: str = message["content"]
                    conversations.append({"from": "human", "value": value})
                elif role=="assistant":
                    if "tool_calls" in message and "content" not in message:
                        value_dict: List[Dict] = message["tool_calls"]
                        train_value = train_format(value_dict)
                        value = f"<tool_call>\n{json.dumps(train_value, ensure_ascii=False)}\n</tool_call>"
                        conversations.append({"from": "gpt", "value": value})
                    elif "content" in message and "tool_calls" not in message:
                        value: str = message["content"]
                        conversations.append({"from": "gpt", "value": value})
                elif role=="tool":
                    value: str = f"<tool_response>\n{message['content']}\n</tool_response>"
                    conversations.append({"from": "human", "value": value})
                        
            training_sample = {
                "id": data_id,
                "system": system_prompt,
                "tools": train_tools,
                "conversations": conversations,
                "chosen": None,
                "rejected": None
            }

            # ----------------------------------------------------
            # 情况 1: 模型正确 (Flag = True) -> 构造合成负例
            # ----------------------------------------------------
            way = 0
            if flag:
                # Chosen: 原始模型的正确回答
                # 注意：如果 answer 是 list (tools)，需要转为 XML 字符串
                if isinstance(answer, list) and isinstance(ground_truth, list):
                    way = 1
                    error_type, error_def = get_error("list")
                    ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)
                    positive_prompt = POSITIVE_COT_TOOL.format(
                        messages_str=messages_str,
                        ground_truth_str=ground_truth_str,
                        reasoning=reasoning,
                        error_type=error_type,
                        error_def=error_def) + POSITIVE_COT_TOOL_OUTPUT
                    response = generate_dpo_tool(client, positive_prompt)
                    if response is None:
                        continue
                    try:
                        try:
                            response = json.loads(response)
                        except Exception:
                            response = get_json(response)
                            if response is None:
                                continue
                    except json.JSONDecodeError:
                        print(f"Skipping ID {data_id}: Invalid JSON received -> {response[:50]}...")
                        continue
                    
                    chosen_cot = response["chosen_cot"]
                    chosen_val = f"<think>\n{chosen_cot}\n</think>\n\n<tool_call>\n{ground_truth_str}\n</tool_call>"
                    rejected_cot = response["rejected_cot"]
                    rejected_tool_call = response["rejected_tool_call"]
                    rejected_val = f"<think>\n{rejected_cot}\n</think>\n\n<tool_call>\n{rejected_tool_call}\n</tool_call>" 
                elif isinstance(answer, str) and isinstance(ground_truth, str):
                    way = 2
                    error_type, error_def = get_error("str")
                    positive_prompt = POSITIVE_COT_STR.format(
                        messages_str=messages_str,
                        ground_truth=ground_truth,
                        answer=answer,
                        reasoning=reasoning,
                        error_type=error_type,
                        error_def=error_def)
                    response = generate_dpo_str(client, positive_prompt)
                    if response is None:
                        continue
                    try:
                        try:
                            response = json.loads(response)
                        except Exception:
                            response = get_json(response)
                            if response is None:
                                continue
                    except json.JSONDecodeError:
                        print(f"Skipping ID {data_id}: Invalid JSON received -> {response[:50]}...")
                        continue

                    chosen_val = response["chosen"]
                    rejected_val = response["rejected"]
                else:
                    continue

            # ----------------------------------------------------
            # 情况 2: 模型错误 (Flag = False) 
            # ----------------------------------------------------
            else:
                differences: str = json.dumps(item.get("differences", []), ensure_ascii=False)
                if isinstance(answer, list) and isinstance(ground_truth, list):
                    ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)
                    answer_str = json.dumps(answer, ensure_ascii=False)
                    judge_tool_prompt = JUDGE_PROMPT_TOOL.format(
                        ground_truth_str=ground_truth_str,
                        answer=answer_str,
                        differences=differences,
                        train_tools=train_tools,
                        reasoning=reasoning)
                    judge_response = judge_tool(client, judge_tool_prompt)
                    if judge_response is None:
                        continue
                    try:
                        try:
                            judge_response = json.loads(judge_response)
                        except Exception:
                            judge_response = get_json(judge_response)
                            if judge_response is None:
                                continue
                    except json.JSONDecodeError:
                        print(f"Skipping ID {data_id}: Invalid JSON received -> {judge_response[:50]}...")
                        continue

                    is_consistent = judge_response["is_consistent"]
                    positive_cot = judge_response["reason"]

                    if is_consistent:
                        way = 3
                        error_type, error_def = get_error("list")
                        negative_tool = NEGATIVE_TOOL.format(
                            messages_str=messages_str,
                            ground_truth_str=ground_truth_str,
                            error_type=error_type,
                            error_def=error_def
                        ) + NEGATIVE_TOOL_OUTPUT
                        negative_response = generate_rejected_tool(client, negative_tool)
                        if negative_response is None:
                            continue
                        try:
                            try:
                                negative_response = json.loads(negative_response)
                            except Exception:
                                negative_response = get_json(negative_response)
                                if negative_response is None:
                                    continue
                        except json.JSONDecodeError:
                            print(f"Skipping ID {data_id}: Invalid JSON received -> {negative_response[:50]}...")
                            continue

                        rejected_cot = negative_response["rejected_cot"]
                        rejected_tool_call = negative_response["rejected_tool_call"]
                    else:
                        way = 4
                        rejected_cot = reasoning
                        rejected_tool_call = answer_str

                    chosen_val = f"<think>\n{positive_cot}\n</think>\n\n<tool_call>\n{ground_truth_str}\n</tool_call>"
                    rejected_val = f"<think>\n{rejected_cot}\n</think>\n\n<tool_call>\n{rejected_tool_call}\n</tool_call>" 
                
                elif isinstance(answer, str) and isinstance(ground_truth, str):
                    judge_str_prompt = JUDGE_PROMPT_STR.format(
                        ground_truth=ground_truth,
                        answer=answer,
                        messages_str=messages_str,
                        reasoning=reasoning)
                    judge_response = judge_str(client, judge_str_prompt)
                    if judge_response is None:
                        continue
                    judge_response = json.loads(judge_response)

                    is_consistent = judge_response["is_consistent"]
                    positive_cot = judge_response["reason"]

                    if is_consistent:
                        way = 5
                        error_type, error_def = get_error("str")
                        negative_str = NEGATIVE_STR.format(
                            messages_str=messages_str,
                            ground_truth=ground_truth,
                            error_type=error_type,
                            error_def=error_def
                        )
                        negative_response = generate_rejected_str(client, negative_str)
                        if negative_response is None:
                            continue
                        try:
                            try:
                                negative_response = json.loads(negative_response)
                            except Exception:
                                negative_response = get_json(negative_response)
                                if negative_response is None:
                                    continue
                        except json.JSONDecodeError:
                            print(f"Skipping ID {data_id}: Invalid JSON received -> {negative_response[:50]}...")
                            continue

                        rejected_val = negative_response["rejected"]
                    else:
                        way = 6
                        rejected_cot = reasoning
                        rejected_tool_call = answer_str
                        rejected_val = f"<think>\n{rejected_cot}\n</think>\n\n{rejected_tool_call}" 

                    chosen_val = f"<think>\n{positive_cot}\n</think>\n\n<tool_call>\n{ground_truth}\n</tool_call>"
                else:
                    if isinstance(ground_truth, list):
                        way = 7
                        ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)
                        generate_cot_tool_prompt = GENERATE_COT_TOOL.format(
                            messages_str = messages_str,
                            ground_truth_str = ground_truth_str
                        )
                        response = generate_cot(client, generate_cot_tool_prompt)
                        if response is None:
                            continue
                        positive_cot = response

                        chosen_val = f"<think>\n{positive_cot}\n</think>\n\n<tool_call>\n{ground_truth_str}\n</tool_call>"
                        rejected_val = f"<think>\n{reasoning}\n</think>\n\n{answer}" 
                    elif isinstance(ground_truth, str):
                        answer_str = json.dumps(answer, ensure_ascii=False)
                        if ground_truth=="":
                            way = 8
                            chosen_val = f"<think>\n\n</think>"
                            rejected_val = f"<think>\n{reasoning}\n</think>\n\n{answer_str}"
                        else:
                            way = 9
                            generate_cot_str_prompt = GENERATE_COT_STR.format(
                                messages_str=messages_str,
                                ground_truth=ground_truth
                            )
                            response = generate_cot(client, generate_cot_str_prompt)
                            if response is None:
                                continue
                            positive_cot = response
                            chosen_val = f"<think>\n{positive_cot}\n</think>\n\n{ground_truth}"
                            rejected_val = f"<think>\n{reasoning}\n</think>\n\n{answer_str}"
                    else:
                        continue

            training_sample["chosen"] = {"from": "gpt", "value": chosen_val}
            training_sample["rejected"] = {"from": "gpt", "value": rejected_val}
            final_dataset.append(training_sample)
            print(f"No.{data_id} chooses way {way}")
            f.write(json.dumps(training_sample, ensure_ascii=False))
            f.write('\n')  # 每行后面加换行符
            f.flush()

    print(f"处理完成，生成 {len(final_dataset)} 条数据，保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    build_dataset()