import json

if __name__=="__main__":
    input_path = "/Users/tankling/all_my_files/coding/qwen/inference/dpo_training.json"
    output_path = "/Users/tankling/all_my_files/coding/qwen/inference/complexFunc.json"

    raw_data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line)
            raw_data_list.append(line_json)
    
    new_data_list = []
    cot = ["reasoning", "refined_cot", "coT", "cot", "refined_coherent_cot", "refined_cot", "CoT", "refined_coT", "flawed_cot", "refined_co_t", "refined_coot", "refined_co_t"]
    for raw_data in raw_data_list:
        new_data = {}
        for key, value in raw_data.items():
            if key=="chosen" and isinstance(value["value"], dict):
                value["value"] = "<think>\n\n</think>\n\n"
            elif key=="rejected" and isinstance(value["value"], dict):
                for kk, vv in value["value"].items():
                    if "answer" not in kk:
                        reasoning = vv
                    else:
                        answer = vv
                rejected_value = f"<think>\n{reasoning}\n</think>\n\n{answer}"
                value["value"] = rejected_value
            if key=="id":
                continue
            new_data[key] = value
        new_data_list.append(new_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data_list, f, ensure_ascii=False, indent=4)