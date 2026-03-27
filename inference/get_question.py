from typing import Dict, List
import json

def rebuild(data: List[Dict]) -> tuple[List[Dict]]:
    output = []
    answer = []
    count = 0
    for idx, record in enumerate(data):
        conversations: List[Dict] = record["conversations"]
        tools: List[Dict] = record["functions"]

        # 调整tools结构
        new_tools: List[Dict] = []
        for func in tools:
            new_tools.append({"type": "function", "function": func})
        
        # 添加 System Prompt
        system_prompt: str = "You are an expert in using functions (i.e., tools) to solve users' tasks. For fountion calling, you need to provide the function name and its arguments. The function name must be same as its name in tool list, and the arguments must obey the format required by the function. Enclose the function call within the tag \"<tool_call></tool_call>\". If possible, you can call multiple functions in parallel, be sure the functions called parallelly are independent of each other."

        messages: List[Dict] = []
        messages.append({"role": "system", "content": system_prompt})
        # 添加每轮的问题
        id = "chatcmpl-"
        for step, turn in enumerate(conversations):
            role = turn["role"]
            if step%2==0:
                value = turn["content"]
                if role=="observation":
                    role = "tool"
                    value = json.dumps(value, ensure_ascii=False)
                messages.append({"role": role, "content": value})
                new_messages = messages.copy()
                output.append({"messages": new_messages, "tools": json.dumps(new_tools, ensure_ascii=False), "id": count})
                count+=1
            else:
                if "content" not in turn and "function_call" in turn:
                    id = "chatcmpl-" + str(count-1)
                    value: List[Dict] = []
                    for func in turn["function_call"]:
                        func_name = func["name"]
                        func_args = func["arguments"]
                        new_func = {"name": func_name, "arguments": json.dumps(func_args, ensure_ascii=False)}
                        value.append({"type": "function",
                                      "function": new_func,
                                      "id": id})
                    messages.append({"role": "assistant", "tool_calls": value})
                elif "content" in turn:
                    value = turn["content"]
                    messages.append({"role": "assistant", "content": value})
                answer.append({"id": count-1, "answer": value})
            
    
    return output, answer

if __name__=="__main__":
    with open("/Users/tankling/all_my_files/coding/inference/ComplexFuncBench.jsonl", 'r', encoding="utf-8") as f:
        data: List[Dict] = []
        for line in f:
            l = json.loads(line)
            data.append(l)

    output, answer = rebuild(data)
    
    # 形成推理格式
    with open("/Users/tankling/all_my_files/coding/qwen/inference/inference_question.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    # 推理的ground truth
    with open("/Users/tankling/all_my_files/coding/qwen/inference/possible_answer.json", 'w', encoding='utf-8') as f:
        json.dump(answer, f, ensure_ascii=False, indent=2)