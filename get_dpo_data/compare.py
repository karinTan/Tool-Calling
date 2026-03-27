import json
from typing import List, Dict
import os

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

def normalize_diff(ground_answer: List[Dict], infer_answer: List[Dict]):
    """
    比较两个函数调用列表是否一致，忽略函数调用的顺序
    
    函数的核心逻辑是将每个函数调用转换为一个标准化的、可哈希的表示，
    然后通过集合比较来判断两个列表是否包含完全相同的函数调用。
    
    标准化转换过程：
    1. 提取函数名称
    2. 将arguments字符串解析为JSON字典
    3. 对参数字典的键值对按key排序，生成标准化的元组表示
    4. 组合函数名和参数元组，形成函数调用的唯一标识
    
    这样转换后，即使：
    - 参数顺序不同（如{"a":1,"b":2}和{"b":2,"a":1}）
    - 函数调用在列表中的顺序不同
    也能被正确识别为相同。
    
    算法步骤详解：
    1. 列表长度检查：如果长度不同，直接返回不一致
    2. 数据标准化：
       - 遍历每个列表，为每个函数调用生成标准化元组 (func_name, args_tuple)
       - args_tuple = tuple(sorted(args_dict.items())) 确保参数顺序无关
    3. 集合比较：
       - 将标准化后的元组列表转换为集合
       - 比较两个集合是否相等
    4. 差异分析：
       - 如果集合不等，找出多出的和缺少的函数调用
       - 检查重复次数是否一致（使用Counter）
    
    示例：
    list1: [{"name":"f1", "args":"{\"a\":1,\"b\":2}"}, 
            {"name":"f1", "args":"{\"c\":3,\"d\":4}"}]
    list2: [{"name":"f1", "args":"{\"b\":2,\"a\":1}"},  # 参数顺序不同
            {"name":"f1", "args":"{\"d\":4,\"c\":3}"}]  # 参数顺序不同
    
    转换后：
    list1_funcs: [('f1', (('a',1),('b',2))), ('f1', (('c',3),('d',4)))]
    list2_funcs: [('f1', (('a',1),('b',2))), ('f1', (('c',3),('d',4)))]
    
    集合比较：set(list1_funcs) == set(list2_funcs) → True
    
    Args:
        list1: 第一个函数调用列表，每个元素需包含function.name和function.arguments
        list2: 第二个函数调用列表，格式同list1
        
    Returns:
        Tuple[bool, List[str]]: 
            - 第一个元素：是否一致（True/False）
            - 第二个元素：具体的差异信息列表，如果一致则为空列表
            
    Raises:
        无显式抛出异常，但会捕获JSON解析错误和键缺失错误，并在差异信息中报告
    """
    differences = []
    
    # 检查列表长度
    if len(ground_answer) != len(infer_answer):
        differences.append(f"The number of functions is different: there are {len(ground_answer)} functions in ground_answer, while infer_answer has {len(infer_answer)}.")
        return False, differences
    
    # 辅助函数：将参数值转换为可哈希的类型
    def make_hashable(value):
        if isinstance(value, list):
            # 递归处理列表中的元素
            return tuple(make_hashable(item) for item in value)
        elif isinstance(value, dict):
            # 递归处理字典中的值
            return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
        else:
            return value
    
    # 为每个函数调用创建规范化表示
    ground_answer_funcs = []
    infer_answer_funcs = []
    
    for i, func in enumerate(ground_answer):
        try:
            func_name = func["function"]["name"]
            args_str = func["function"]["arguments"]
            args_dict = json.loads(args_str)
            
            # 处理参数值，确保所有值都是可哈希的
            # hashable_args = []
            # for key, value in sorted(args_dict.items()):
                # hashable_value = make_hashable(value)
                # hashable_args.append((key, hashable_value))
            
            # 创建可哈希的表示
            # sorted_args = tuple(hashable_args)
            sorted_args = dict(sorted(args_dict.items()))
            ground_answer_funcs.append((func_name, sorted_args))
        except (KeyError, json.JSONDecodeError) as e:
            differences.append(f"No.{i+1} of ground_answer is wrong: {e}")
            return False, differences
    
    for i, func in enumerate(infer_answer):
        try:
            func_name = func["function"]["name"]
            args_str = func["function"]["arguments"]
            args_dict = json.loads(args_str)
            
            # 处理参数值，确保所有值都是可哈希的
            # hashable_args = []
            # for key, value in sorted(args_dict.items()):
                # hashable_value = make_hashable(value)
                # hashable_args.append((key, hashable_value))
            
            # 创建可哈希的表示
            # sorted_args = tuple(hashable_args)
            sorted_args = dict(sorted(args_dict.items()))
            infer_answer_funcs.append((func_name, sorted_args))
        except (KeyError, json.JSONDecodeError) as e:
            differences.append(f"No.{i+1} of infer_answer is wrong: {e}")
            return False, differences
    
        # 比较每个函数调用
    all_match = True
    
    # 先按顺序尝试匹配
    for i, (norm1, norm2) in enumerate(zip(ground_answer_funcs, infer_answer_funcs)):
        if norm1 is None or norm2 is None:
            all_match = False
            continue
            
        name1, args1 = norm1
        name2, args2 = norm2
        
        # 检查函数名
        if name1 != name2:
            differences.append(f"The name of no.{i+1} function is different: '{name1}' vs '{name2}'")
            all_match = False
            continue
        
        # 检查参数数量
        if len(args1) != len(args2):
            differences.append(f"The number of arguments of no.{i+1} function is different: {len(args1)} vs {len(args2)}. Arguments in ground_answer: {list(args1.keys())}. Arguments in infer_answer: {list(args2.keys())}")
            all_match = False
            continue
        
        # 检查每个参数的值
        for key in args1:
            if key not in args2:
                differences.append(f"Missing argument '{key}' in no.{i+1} function '{name1}'.")
                all_match = False
                continue
            if args1[key] != args2[key]:
                differences.append(f"The value of argument '{key}' in no.{i+1} function '{name1}' is different: {args1[key]} vs {args2[key]}")
                all_match = False
        
        # 检查list2中是否有list1没有的参数
        for key in args2:
            if key not in args1:
                differences.append(f"Redundant argument '{key}' in no.{i+1} function '{name1}'")
                all_match = False
    
    return all_match, differences

def diff(truth_list: List[Dict], infered_list: List[Dict]):
    comparison_list = []
    for idx in range(len(truth_list)):
        ground_truth = truth_list[idx]
        infered = infered_list[idx]

        ground_id = ground_truth["id"]
        ground_answer = ground_truth["answer"]
        if isinstance(ground_answer, list):
            train_ground_answer = train_format(ground_answer)
        else:
            train_ground_answer = ground_answer

        infer_id = infered["id"]
        infer_answer = ""
        reasoning = ""
        if "reasoning" in infered:
            reasoning = infered["reasoning"]
        if "tool_calls" in infered and len(infered["tool_calls"])!=0:
            infer_answer = infered["tool_calls"]
            train_infer_answer = train_format(infer_answer)
        else:
            infer_answer = infered["content"]
            train_infer_answer = infer_answer
        
        comparison_item = {}
        differences = []
        if ground_id==infer_id:
            if isinstance(ground_answer, str) and isinstance(infer_answer, str):
                if ground_answer==infer_answer or ground_answer=="":
                    flag = True
                else:
                    flag = False
                    differences.append(ground_answer)
                    differences.append(infer_answer)
                comparison_item = {
                    "id": infer_id,
                    "ground_truth": train_ground_answer,
                    "answer": train_infer_answer,
                    "reasoning": reasoning,
                    "flag": flag,
                    "differences": differences
                }
            elif isinstance(ground_answer, list) and isinstance(infer_answer, list):
                flag, differences = normalize_diff(ground_answer, infer_answer)
                comparison_item = {
                    "id": infer_id,
                    "ground_truth": train_ground_answer,
                    "answer": train_infer_answer,
                    "reasoning": reasoning,
                    "flag": flag,
                    "differences": differences
                }
            else:
                flag = False
                differences.append(ground_answer)
                differences.append(infer_answer)
                comparison_item = {
                    "id": infer_id,
                    "ground_truth": train_ground_answer,
                    "answer": train_infer_answer,
                    "reasoning": reasoning,
                    "flag": flag,
                    "differences": differences
                }
        else:
            comparison_item = {
                "id": ground_id,
                "ground_truth": train_ground_answer,
                "answer": "Not found",
                "reasoning": "Not found",
                "flag": False,
                "differences": [train_ground_answer, "Not found"]
            }
        
        comparison_list.append(comparison_item)
    
    return comparison_list
        
def uninfered(ground_truth: List[Dict], infered: List[Dict], new_infered_path: str):
    last_id = infered[-1]["id"]
    new_ground_truth = ground_truth[:last_id+1]
    new_infered = []
    idx = 0
    with open(new_infered_path, 'w', encoding='utf-8') as f:
        for data in infered:
            id = data["id"]
            while(id>idx):
                new_data = {
                    "id": idx,
                    "role": "assistant",
                    "content": "Not found",
                    "tool_calls": [],
                    "raw_response": {},
                    "reasoning": "Not found"
                }
                new_infered.append(new_data)
                f.write(json.dumps(new_data, ensure_ascii=False))
                f.write('\n')  # 每行后面加换行符
                idx+=1

            new_infered.append(data)
            f.write(json.dumps(data, ensure_ascii=False))
            f.write('\n')  # 每行后面加换行符
            idx+=1

    return new_ground_truth, new_infered

if __name__=="__main__":
    ground_truth_path = "/Users/tankling/Documents/all_my_files/coding/qwen/inference/possible_answer.json"
    infered_path = "/Users/tankling/Documents/all_my_files/coding/qwen/inference/inference_answer.json"
    
    ground_truth: List[Dict] = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    infered: List[Dict] = []
    with open(infered_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                piece = json.loads(line)
                infered.append(piece)
            except json.JSONDecodeError as e:
                print(f"\n❌ 第 {line_number} 行解析失败:")
                print(f"   错误信息: {e}")

    dirname = os.path.dirname(infered_path)
    new_infered_path = dirname + "/complete_inference_answer.json"
    new_ground_truth, new_infered = uninfered(ground_truth, infered, new_infered_path)
    
    print(len(new_infered), len(new_ground_truth))
    assert len(new_infered)==len(new_ground_truth)

    comparison_list = diff(new_ground_truth, new_infered)
    comparison_path = dirname + "/comparison.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        for comparison_item in comparison_list:
            f.write(json.dumps(comparison_item, ensure_ascii=False))
            f.write('\n')  # 每行后面加换行符
