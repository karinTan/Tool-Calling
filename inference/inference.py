import json
import os
from typing import List, Dict
from openai import OpenAI
import re
from tqdm import tqdm

# ================= 配置区域 =================
API_BASE = "http://localhost:8000/v1"
API_KEY = ""
MODEL_NAME = "qwen3-4b-think-mine"

# ================= 主推理逻辑 =================

def _inference(data_entry: Dict, client):
    # 1. 准备 Tools
    tools = json.loads(data_entry.get('tools', "[]"))
    id = data_entry.get("id", -1)
    # 2. 准备 Messages (历史上下文)
    messages = data_entry.get("messages", [])
    print(f"发送给模型的消息数量: {len(messages)}")

    try:
        # 3. 发送请求
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=8192,
            n=1,
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                # 开启深度思考
                "chat_template_kwargs": {"enable_thinking": True},
                # "enable_thinking": True,
                "top_k": 20,
            },
            tools=tools,
            parallel_tool_calls=True,
            stream=False,
            tool_choice="auto"
        )

        response_dict = response.to_dict()
        reasoning = getattr(response.choices[0].message, "reasoning", "")
        # reasoning = response.choices[0].message.reasoning
        raw_content = response.choices[0].message.content

        # 1. 定义正则提取所有的 tool_call 块
        tool_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(tool_pattern, raw_content, re.DOTALL)

        # 2. 构造符合你要求的 assistant 消息格式
        if matches:
            tool_calls_list = []
            for index, match_str in enumerate(matches):
                try:
                    tool_data = json.loads(match_str.strip())
                    
                    # 构造单个 tool_call 项
                    tool_call_item = {
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name"),
                            # 确保 arguments 是字符串格式的 JSON
                            "arguments": json.dumps(tool_data.get("arguments"), ensure_ascii=False)
                        },
                        "id": f"chatcmpl-{id}"
                    }
                    tool_calls_list.append(tool_call_item)
                except json.JSONDecodeError:
                    continue

            # 3. 最终组合成你需要的 assistant 字典
            assistant_message = {
                "id": id,
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls_list,
                "raw_response": response_dict
            }
            
            # 如果有思维链内容，可以按需存储
            if reasoning:
                assistant_message["reasoning"] = reasoning

        else:
            assistant_message = {
                "id": id,
                "role": "assistant",
                "content": raw_content,
                "tool_calls": [],
                "raw_response": response_dict
            }
            
            # 如果有思维链内容，可以按需存储
            if reasoning:
                assistant_message["reasoning"] = reasoning

        return assistant_message

    except Exception as e:
        print(f"Inference Error: {e}")
        return None

def run_inference(input_path: str, output_path: str):
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
     # 逐行写入（JSON Lines 格式）
    with open(output_path, 'w', encoding='utf-8') as f:
        for data_entry in tqdm(data_list, desc="推理进度", unit="条"):
            assistant_message = _inference(data_entry, client)
            
            # 每行一个完整的 JSON 对象
            if assistant_message:
                f.write(json.dumps(assistant_message, ensure_ascii=False))
                f.write('\n')  # 每行后面加换行符
# ================= 运行示例 =================

if __name__ == "__main__":
    input_path = "/root/autodl-tmp/zhipu/inference_question.json"
    dirname = os.path.dirname(input_path)
    output_path = dirname + "/inference_answer.json"
   
    run_inference(input_path, output_path)