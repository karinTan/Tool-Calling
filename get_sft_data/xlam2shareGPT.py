import json
import os
from typing import Dict, Any

def transform2sharegpt(input_file: str, output_file: str):
    """
    将原始格式转换为当前训练使用的 sharegpt 风格记录
    
    原始格式示例:
    [
      {
        "id": 0,
        "query": "Where can I find live giveaways for beta access and games?",
        "answers": "[{\"name\": \"live_giveaways_by_type\", \"arguments\": {\"type\": \"beta\"}}, ...]",
        "tools": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways...\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways...\", \"type\": \"str\", \"default\": \"game\"}}}]"
      }
    ]
    
    目标格式:
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "", "function_call": {...}}
      ],
      "class": "simple",
      "functions": [...]
    }
    """
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"找到 {len(raw_data)} 条数据")
    
    # 准备转换后的数据列表
    converted_data = []
    
    for i, item in enumerate(raw_data):
        # 解析工具和答案
        try:
            tools_list = json.loads(item.get("tools", []))
            answers_list = json.loads(item.get("answers", []))
            query = item.get("query", "")
            
            # 创建 messages 数组
            messages = []
            
            # 1. 添加用户消息
            messages.append({
                "from": "human",
                "value": query
            })
            
            value = []
            # 2. 添加函数调用消息
            for answer in answers_list:
                function_name = answer.get("name", "")
                arguments = answer.get("arguments", {})

                value.append({
                    "name": function_name,
                    "arguments": arguments
                })
                
            messages.append({
                "from": "function_call",
                "value": json.dumps(value, ensure_ascii=False)
            })
            
            # 3. 转换工具定义格式
            functions = []
            for tool in tools_list:
                function_def = normalize_tool_definition(tool)
                if function_def:
                    functions.append(function_def)

            # 4. 分类
            category = ""
            if len(tools_list)==1 and len(answers_list)==1:
                category = "simple"
            elif len(tools_list)==1 and len(answers_list)>1:
                category = "parallel"
            elif len(tools_list)>1 and len(answers_list)==1:
                category = "multiple"
            elif len(tools_list)>1 and len(answers_list)>1:
                category = "parallel multiple"
            else:
                print("wrong class")

            # 5. 创建最终数据项
            if functions and messages and category:
                converted_item = {
                    "conversations": messages,
                    "category": category,
                    "tools": json.dumps(functions, ensure_ascii=False)
                }
                converted_data.append(converted_item)
            
        except json.JSONDecodeError as e:
            print(f"警告: 第 {i+1} 条数据 JSON 解析失败: {e}")
            continue
        except Exception as e:
            print(f"警告: 处理第 {i+1} 条数据时出错: {e}")
            continue
    
    print(f"成功转换 {len(converted_data)} 条数据")
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成! 结果保存到: {output_file}")
    
    return converted_data

def normalize_tool_definition(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    将原始工具定义转换为统一格式
    
    原始格式:
    {
      "name": "live_giveaways_by_type",
      "description": "Retrieve live giveaways...",
      "parameters": {
        "type": {
          "description": "The type of giveaways...",
          "type": "str",
          "default": "game"
        }
      }
    }
    
    目标格式:
    {
      "name": "live_giveaways_by_type",
      "description": "Retrieve live giveaways...",
      "parameters": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "description": "The type of giveaways...",
            "default": "..." # optional
          }
        },
        "required": ["type"]
      }
    }
    """
    
    try:
        function_name = tool.get("name", "")
        description = tool.get("description", "")
        raw_params = tool.get("parameters", {})
        
        # 转换参数格式
        properties = {}
        required = []
        
        for param_name, param_info in raw_params.items():
            param_type = param_info.get("type", "")
            # 映射 Python 类型到 JSON 类型
            param_type_list = param_type.split(",")

            if len(param_type_list) > 1:
                json_type = infer_json_type_from_type_hint(param_type_list[0].lower())
            else:
                json_type = infer_json_type_from_type_hint(param_type.lower())
                required.append(param_name)

            param_def = {
                "type": json_type,
                "description": param_info.get("description", "")
            }
            
            properties[param_name] = param_def
        
        return {
            "name": function_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required if required else []  # 至少需要一个参数
            }
        }
    
    except Exception as e:
        print(f"工具转换失败: {e}")
        return None


def infer_json_type_from_type_hint(type_str: str) -> str:
    """
    从类型字符串中提取 JSON 类型（模糊匹配）

    示例:
    "str" -> "string"
    "string, optional" -> "string"
    "List[str]" -> "array"
    "Dict[str, any]" -> "object"
    "int | None" -> "integer"
    """
    if not isinstance(type_str, str):
        return "string"

    type_str_lower = type_str.lower().strip()

    # 定义关键词映射（只要包含关键词就匹配）
    keyword_mapping = {
        # 字符串类型
        "str": "string",
        "string": "string",
        "text": "string",
        "char": "string",

        # 数字类型
        "int": "integer",
        "integer": "integer",
        "number": "number",
        "float": "number",
        "double": "number",
        "decimal": "number",

        # 布尔类型
        "bool": "boolean",
        "boolean": "boolean",

        # 数组类型
        "list": "array",
        "array": "array",
        "[]": "array",  # 处理类似 "List[str]" 的情况

        # 对象类型
        "dict": "object",
        "object": "object",
        "map": "object",
        "{}": "object",  # 处理类似 "Dict[str, any]" 的情况
    }

    # 特殊处理组合类型
    if "|" in type_str_lower:
        # 处理 "int | None" 或 "str | None" 等情况
        parts = [part.strip() for part in type_str_lower.split("|")]
        for part in parts:
            if part != "none" and part != "null":
                return infer_json_type_from_type_hint(part)
        return "string"

    # 处理泛型类型，如 List[str], Dict[str, any]
    if "list[" in type_str_lower or "array[" in type_str_lower:
        return "array"
    if "dict[" in type_str_lower or "object[" in type_str_lower:
        return "object"

    # 模糊匹配关键词
    for keyword, json_type in keyword_mapping.items():
        if keyword in type_str_lower:
            return json_type

    # 默认返回 string
    return "string"

def contains_none_recursive(data):
    """
    递归检查JSON数据中是否存在None值
    返回True表示存在None，False表示不存在None
    """
    if data is None:
        return True

    if isinstance(data, dict):
        for key, value in data.items():
            if contains_none_recursive(value):
                print(key)
                return True
        return False

    elif isinstance(data, list):
        for item in data:
            if contains_none_recursive(item):
                return True
        return False

    return False

if __name__ == "__main__":
    # 配置输入输出文件
    input_file = "./xlam_function_calling_60k.json"  # 修改为你的输入文件
    transformed_output_file = "./transformed_xlam.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")

    print("\n1. 转换为 sharegpt 格式...")
    transformed_data = transform2sharegpt(input_file, transformed_output_file)

    if contains_none_recursive(transformed_data):
       print("wrong")

    # 显示转换统计信息
    if transformed_data:
        print(f"\n转换统计:")
        print(f"- 总数据量: {len(transformed_data)}")
        
        # 统计函数调用次数
        total_function_calls = 0
        function_types = {}
        
        for item in transformed_data:
            messages = item.get("messages", [])
            for msg in messages:
                if msg.get("role", "") == "assistant" and "function_call" in msg:
                    total_function_calls += 1
                    func_name = msg["function_call"]["name"]
                    function_types[func_name] = function_types.get(func_name, 0) + 1
        
        print(f"- 总函数调用次数: {total_function_calls}")
        # print(f"- 函数类型分布:")
        # for func_name, count in function_types.items():
        #     print(f"  - {func_name}: {count} 次")