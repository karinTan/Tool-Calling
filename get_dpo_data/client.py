# ================= 配置区域 =================
TEACHER_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEACHER_API_KEY = "sk-336804f6c9144365b4783cf362449182" 
MODEL_NAME = "qwen-plus-2025-07-14" 

def generate_dpo_tool(client, prompt):
    """
    Teacher 模式：根据 Correct Action 倒推思考过程。
    """
    try:
        system = "You are an expert AI data annotator. Your task is to refine agent reasoning and construct DPO (Direct Preference Optimization) data pairs."
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "dpo_pair_with_cot",
                    "description": "DPO preference pair with chosen and rejected reasoning and tool calls",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "chosen_cot": {
                                "type": "string",
                                "description": "Correct concise reasoning."
                            },
                            "rejected_cot": {
                                "type": "string",
                                "description": "Flawed reasoning."
                            },
                            "rejected_tool_call": {
                                "type": "string",
                                "description": "Incorrect tool calls. Format: [{\"name\": \"function_name1\", \"arguments\": {\"argument1\": \"value1\"}}]"
                            }
                        },
                        "additionalProperties": False,
                        "required": ["chosen_cot", "rejected_cot", "rejected_tool_call"]
                    }
                }
            }
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            },
            response_format=response_format
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None
    
def generate_dpo_str(client, prompt):
    """
    Teacher 模式：根据 Correct Action 倒推思考过程。
    """
    try:
        system = "You are an expert AI data annotator. Your task is to refine agent reasoning and construct DPO (Direct Preference Optimization) data pairs."
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "dpo_pair_with_cot",
                    "description": "DPO preference pair with chosen and rejected reasoning and answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "chosen": {
                                "type": "string",
                                "description": "Correct concise reasoning wrapped in <think> tags and relevant summarization."
                            },
                            "rejected": {
                                "type": "string",
                                "description": "Flawed reasoning wrapped in <think> tags and irrelevant summarization."
                            }
                        },
                        "additionalProperties": False,
                        "required": ["chosen", "rejected"]
                    }
                }
            }
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            },
            response_format=response_format
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None
    
def judge_tool(client, prompt):
    try:
        system = "You are a specialist in Agent Tool-Use Evaluation and Protocol Debugging."
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_tool_call",
                    "description": "Judge the correctness of tool calls and generate cot.",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_consistent": {
                                "type": "boolean",
                                "description": "Whether the answer is consistent with the ground truth."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Correct concise reasoning."
                            }
                        },
                        "additionalProperties": False,
                        "required": ["is_consistent", "reason"]
                    }
                }
            }
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            },
            response_format=response_format
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None
    
def generate_rejected_tool(client, prompt):
    try:
        system = "You are an expert AI data annotator. Your task is to construct rejected data for DPO (Direct Preference Optimization)."
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "generate_rejected_tool_call",
                    "description": "DPO preference pair with rejected reasoning and answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rejected_cot": {
                                "type": "string",
                                "description": "Flawed reasoning."
                            },
                            "rejected_tool_call": {
                                "type": "string",
                                "description": "Incorrect tool calls. Format: [{\"name\": \"function_name1\", \"arguments\": {\"argument1\": \"value1\"}}]"
                            }
                        },
                        "additionalProperties": False,
                        "required": ["rejected_cot", "rejected_tool_call"]
                    }
                }
            }
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            },
            response_format=response_format
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None


def judge_str(client, prompt):
    try:
        system = "You are a specialist in Agent Tool-Use Summarization Evaluation."
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_str",
                    "description": "Judge the correctness of summarization and generate cot.",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_consistent": {
                                "type": "boolean",
                                "description": "Whether the summarization has the same semantic information with the ground truth and conversation history."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Correct concise reasoning."
                            }
                        },
                        "additionalProperties": False,
                        "required": ["is_consistent", "reason"]
                    }
                }
            }
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            },
            response_format=response_format
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None

def generate_rejected_str(client, prompt):
    try:
        system = "You are an expert AI data annotator. Your task is to construct rejected data for DPO (Direct Preference Optimization)."
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "generate_rejected_str",
                    "description": "DPO preference pair with rejected reasoning and answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rejected": {
                                "type": "string",
                                "description": "Flawed reasoning and answer."
                            }
                        },
                        "additionalProperties": False,
                        "required": ["rejected"]
                    }
                }
            }
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            },
            response_format=response_format
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None

def generate_cot(client, prompt):
    try:
        system = "You are an expert AI data annotator. Your task is to construct agent reasoning."
        response = client.chat.completions.create(
            temperature=0.01,
            top_p=0.95,
            presence_penalty=0.0,
            max_tokens=4096,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False,
                "top_k": 20
            }
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Teacher API Error: {e}")
        return None