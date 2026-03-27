import json
import os
from typing import List, Dict, Any

def add_cot_to_conversations(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    为转换后的数据添加 CoT（思考过程）
    
    目标格式:
    [
      {
        "from": "human",
        "value": "..."
      },
      {
        "from": "gpt",
        "value": "<think>用think token包裹的cot</think>"
      },
      {
        "from": "observation",
        "value": ""
      },
      {
        "from": "function_call",
        "value": "..."
      }
    ]
    """
    conversations = data_item.get("conversations", [])
    category = data_item.get("category", "")
    
    if len(conversations) < 2:
        return data_item
    
    if category=="simple" or category=="parallel":
        new_data_item = {
            "conversations": conversations,
            "tools": data_item.get("tools", [])
        }
        return new_data_item
    
    # 找到 human 和 function_call 消息
    human_msg = None
    function_call_msg = None
    
    for msg in conversations:
        if msg.get("from") == "human":
            human_msg = msg
        elif msg.get("from") == "function_call":
            function_call_msg = msg
    
    if not human_msg or not function_call_msg:
        return data_item
    
    # 提取用户查询
    user_query = human_msg.get("value", "")
    
    # 解析 function_call 消息
    function_calls = []
    try:
        fc_value = function_call_msg.get("value", "[]")
        fc_data = json.loads(fc_value)
        if isinstance(fc_data, list):
            function_calls = fc_data
        else:
            # 如果不是列表，可能是一个单独的函数调用
            function_calls = [fc_data]
    except:
        # 尝试解析为单一函数调用
        try:
            fc_value = function_call_msg.get("value", "{}")
            fc_data = json.loads(fc_value)
            function_calls = [fc_data]
        except:
            return data_item
    
    # 根据 category 生成 CoT
    cot_content = build_cot_for_category(user_query, function_calls, category)
    
    # 创建新的 conversations 列表
    new_conversations = []
    
    # 1. 添加 human 消息
    new_conversations.append(human_msg)
    
    # 2. 添加 gpt 消息（带 CoT）
    gpt_message = {
        "from": "gpt",
        "value": f"<think>\n{cot_content}\n</think>"
    }
    new_conversations.append(gpt_message)
    
    # 3. 添加 observation 消息（固定为空）
    observation_message = {
        "from": "observation",
        "value": ""
    }
    new_conversations.append(observation_message)
    
    # 4. 添加 function_call 消息（保持不变）
    new_conversations.append(function_call_msg)
    
    # 创建新的数据项
    new_data_item = {
        "conversations": new_conversations,
        "tools": data_item.get("tools", [])
    }
    
    return new_data_item

def build_cot_for_category(user_query: str, function_calls: List[Dict], category: str) -> str:
    """
    根据类别生成不同的 CoT 内容
    """
    # 提取函数名称和参数
    function_info_list = []
    
    for fc in function_calls:
        func_name = fc.get("name", "")
        arguments_str = fc.get("arguments", "{}")
        
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            
            # 格式化参数
            if arguments:
                args_list = []
                for key, value in arguments.items():
                    value_str = str(value)
                    # 如果参数值太长，进行截断
                    if len(value_str) > 20:
                        value_str = value_str[:17] + "..."
                    args_list.append(f"{key}: {value_str}")
                
                args_text = f"({', '.join(args_list)})"
            else:
                args_text = "()"
        except:
            args_text = "()"
        
        function_info_list.append(f"{func_name}{args_text}")
        
    if category == "multiple":
        # 多个函数：多个不同函数，但顺序调用
        functions_str = "、".join(function_info_list)
        return f"分析：需要依次调用多个不同的函数。\n步骤：1. 分解用户需求。2. 确定需要调用的多个函数。3. 按顺序为每个函数提取参数。\n调用：{functions_str}"
    
    elif category == "parallel multiple":
        # 并行多个：多个不同函数，可能并行调用
        functions_str = "、".join(function_info_list)
        return f"分析：需要多次调用多个不同的函数。\n步骤：1. 分解用户需求。2. 确定需要调用的多个函数。3. 分析调用顺序。4. 提取各函数参数。\n调用：{functions_str}"
    
def build_cot_dataset(input_file: str, output_file: str):
    """
    处理已转换的数据，添加 CoT
    """
    # 读取转换后的数据
    with open(input_file, 'r', encoding='utf-8') as f:
        converted_data = json.load(f)
    
    print(f"读取到 {len(converted_data)} 条数据")
    
    # 处理数据
    processed_data = []
    stats = {
        "simple": 0,
        "parallel": 0,
        "multiple": 0,
        "parallel multiple": 0,
        "other": 0
    }
    
    for i, item in enumerate(converted_data):
        category = item.get("category", "").lower()
        
        # 统计
        if category in stats:
            stats[category] += 1
        else:
            stats["other"] += 1
        
        # 添加 CoT
        processed_item = add_cot_to_conversations(item)
        processed_data.append(processed_item)
        
        # 打印进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 条数据")
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n数据统计:")
    print(f"总数据量: {len(processed_data)}")
    for category, count in stats.items():
        if count > 0:
            print(f"{category}: {count} 条 ({count/len(processed_data)*100:.1f}%)")
    
    print(f"\n处理完成! 结果保存到: {output_file}")
    
    return processed_data

if __name__ == "__main__":
    # 配置文件路径
    converted_file = "./transformed_xlam2gpt.json"  # xlam2shareGPT.py 的输出
    cot_output_file = "../xlam_gpt_cot.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(converted_file):
        print(f"错误: 输入文件 {converted_file} 不存在!")
        print("请先运行 transform_xlam2gpt.py 进行转换")
        exit(1)
    
    # 步骤1: 为转换后的数据添加 CoT
    print("步骤1: 为转换后的数据添加 CoT...")
    cot_data = build_cot_dataset(converted_file, cot_output_file)
    
    print("\n所有处理完成!")
    print(f"带 CoT 的数据: {cot_output_file}")