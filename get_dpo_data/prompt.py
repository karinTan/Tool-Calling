# ================= ERROR TYPES DEFINITIONS =================

# for negative result: Notice that when you see redundant parameters from the model response when it is function call, it might because it gives all the parameters even the default ones. So, as long as other parameters take the same values, regard this as correct. In the first line, return yes or no. If your answer is no, in the second line, return a number to represent the error type.

ERROR_TYPES = {
    "Nested function calls": "There are missing function calls. Model fails to call some necessary functions because they are not explicitly mentioned in the query.",
    "Short dependency": "There are outputs from a previous function call in this turn that is not used correctly in later function calls.",
    "Long dependency": "There are some parameter values exist in the conversation history but not properly used in this turn.",
    "Missed function or parameters": "There are some parameter values or functions present or not present in the context while the model thinks the opposite.",
    "Wrong summarization": "Whether the model response is a wrong summarization of the reference response.", # only used when the ground truth is string
    "Partially relevant": "Ignores important outputs from earlier conversation or fails to answer the user's original question."
}
# ================= TEACHER PROMPT TEMPLATE =================

POSITIVE_COT_TOOL = """
### INPUT DATA
- Conversation History: {messages_str}
- Ground Truth Tool Calls (Confirmed Correct): {ground_truth_str}
- Original Model Reasoning: {reasoning}

### TASK 1: CONSTRUCT 'CHOSEN'
The Ground Truth Tool Calls is correct. You must evaluate if the 'Original Model Reasoning' is redundant and rewrite it into a concise & effective CoT.
**Criteria for Redundancy:**
1. Repeating unnecessary tool descriptions or basic definitions.
2. Including polite filler or meta-talk (e.g., "I will now search for...").
3. Narrating the obvious (e.g., "The user asked a question, so I will answer").
**Requirements for Concise CoT:**
1. Focus only on logic: Why these tools? Why these specific parameters?
2. If parameters are derived from history or calculation, show the derivation briefly.
3. Use professional, direct language.

### TASK 2: CONSTRUCT 'REJECTED'
Construct a negative response by injecting a specific error. Error Type to Inject: {error_type}. Definition: {error_def}.
**Requirements:**
1. Change the Ground Truth Tool Calls to reflect this specific error.
2. Write a reasoning block according to the error type, which leads illogically or incorrectly to the wrong tool calls.
"""
POSITIVE_COT_TOOL_OUTPUT = """
### OUTPUT FORMAT
Return a valid JSON object:
{{
  "chosen_cot": "Refined Concise CoT",
  "rejected_cot": "Flawed CoT",
  "rejected_tool_call": "Flawed Tool Calls in format: [{\"name\": \"function_name\", \"arguments\": {\"argument1\": \"value1\"}}]"
}}
"""

POSITIVE_COT_STR = """
### INPUT DATA
- Conversation History: {messages_str}
- Ground Truth Answer: {ground_truth}
- Original Model Answer: {answer}
- Original Model Reasoning: {reasoning}

### TASK 1: CONSTRUCT 'CHOSEN'
The Original Model Answer is considered correct under either of the following conditions:
1. They are identical;
2. The Ground Truth Answer is an empty string, and the Original Model Answer is a reasonable and faithful summary.
**You must refine the CoT:**
1. Remove redundant explanations, meta-talk, and obvious narration.
2. Retain only the minimal logical justification needed to support the final answer.
3. Use professional, direct, and concise language.
**If the ground truth is empty, rewrite the Answer:**
1. Preserve the original meaning and key information of the answer.
2. If it is a summarization, ensure it faithfully reflects the core intent of the conversation.
3. Remove verbosity, repetition, or irrelevant details.
4. Do NOT introduce new information.

### TASK 2: CONSTRUCT 'REJECTED'
Construct a negative response by injecting a specific error. Error Type to Inject: {error_type}. Definition: {error_def}.
**Requirements:**
1. Change the Ground Truth Answer to reflect this specific error.
2. Write a reasoning block according to the error type, which leads illogically or incorrectly to the wrong answer.

### OUTPUT FORMAT
Return a valid JSON object:
{{
  "chosen": "<think>
Refined Concise CoT
</think>

Refined Correct Answer String",
  "rejected": "<think>
Flawed CoT
</think>

Flawed Answer String"
}}
"""


JUDGE_PROMPT_STR = """
### INPUT DATA
- Ground Truth: {ground_truth}
- Model Prediction: {answer}
- Conversation History: {messages_str}
- Original Model Reasoning: {reasoning}

### TASK 1: COMPARE AND JUDGE
Compare the Ground Truth and the Model Prediction. Determine if they convey the same semantic meaning and information, even if phrased differently.
**Requirements:**
1. Analyze and list the key information contained in the Ground Truth.
2. Analyze and list the key information contained in the Model Prediction.
3. Compare the two lists. If the Model Prediction misses key points or hallucinates incorrect details not present in the Conversation History, mark the prediction as WRONG.

### TASK 2: REFINE COT
Based on the judgment in Task 1:
- If the Model Prediction is CORRECT, refine the 'Original Model Reasoning' to be more concise and effective.
- If the Model Prediction is WRONG, discard the original reasoning and construct a NEW, correct Chain-of-Thought (CoT) that logically leads to the Ground Truth.
**Requirements for CoT:**
1. Remove redundant explanations, meta-talk, and obvious narration.
2. Retain only the minimal logical justification needed to support the final answer.
3. Use professional, direct, and concise language.

### OUTPUT FORMAT
Return a valid JSON object:
{{
  "is_consistent": True/False,
  "reason": "Concise CoT"
}}
"""

NEGATIVE_STR = """
### INPUT DATA
- Conversation History: {messages_str}
- Ground Truth Answer: {ground_truth}

### TASK: CONSTRUCT 'REJECTED' SAMPLE
Construct a negative response by injecting a specific error. Error Type to Inject: {error_type}. Definition: {error_def}.
**Requirements:**
1. Change the Ground Truth Answer to reflect this specific error.
2. Write a reasoning block according to the error type, which leads illogically or incorrectly to the wrong answer.

### OUTPUT FORMAT
Return a valid JSON object:
{{
  "rejected": "<think>
Flawed CoT
</think>

Flawed Answer String"
}}
"""

JUDGE_PROMPT_TOOL = """
### INPUT DATA
- Ground Truth Tool Calls: {ground_truth_str}
- Model Prediction: {answer}
- Differences Log: {differences}
- Tools: {train_tools}
- Original Model Reasoning: {reasoning}

### TASK 1: COMPARE AND JUDGE
Compare the Ground Truth Tool Call and the Model Prediction Tool Call. The Model Prediction is marked as "different" by exact match, but it might be effectively correct.
**Criteria for Correctness:**
1. The function name is the same.
2. All parameters in GT are present in Prediction with matching values.
3. Crucial: If Prediction has extra parameters not in GT, check if they are default values in Tools. If yes, count as CORRECT.
4. If Prediction has wrong values for parameters, count as INCORRECT.

### TASK 2: REFINE COT
Based on the judgment in Task 1:
- If the Model Prediction is CORRECT, refine the 'Original Model Reasoning' to be more concise and effective.
- If the Model Prediction is WRONG, discard the original reasoning and construct a NEW, correct Chain-of-Thought (CoT) that logically leads to the Ground Truth.
**Requirements for CoT:**
1. Focus only on logic: Why these tools? Why these specific parameters?
2. Retain only the minimal logical justification needed to support the final answer.
3. Use professional, direct, and concise language.

### OUTPUT FORMAT
Return a valid JSON object:
{{
  "is_consistent": True/False,
  "reason": "Concise CoT"
}}
"""

NEGATIVE_TOOL = """
### INPUT DATA
- Conversation History: {messages_str}
- Ground Truth Tool Calls: {ground_truth_str}

### TASK: CONSTRUCT 'REJECTED' SAMPLE
Construct a negative response by injecting a specific error. Error Type to Inject: {error_type}. Definition: {error_def}.
**Requirements:**
1. Change the Ground Truth Tool Calls to reflect this specific error.
2. Write a reasoning block according to the error type, which leads illogically or incorrectly to the wrong tool calls.
"""
NEGATIVE_TOOL_OUTPUT = """
### OUTPUT FORMAT
Return a valid JSON object:
{{
  "rejected_cot": "Flawed CoT",
  "rejected_tool_call": "Flawed Tool Calls in format: [{\"name\": \"function_name\", \"arguments\": {\"argument1\": \"value1\"}}]"
}}
"""

GENERATE_COT_TOOL = """
### INPUT DATA
- Conversation History: {messages_str}
- Ground Truth Tool Calls (Confirmed Correct): {ground_truth_str}

### TASK: CONSTRUCT COT
Construct a correct Chain-of-Thought (CoT) that logically leads to the Ground Truth.
**Requirements for Concise CoT:**
1. Focus only on logic: Why these tools? Why these specific parameters?
2. If parameters are derived from history or calculation, show the derivation briefly.
3. Use professional, direct, and concise language.
4. No redundant explanations, meta-talk, and obvious narration.
"""

GENERATE_COT_STR = """
### INPUT DATA
- Conversation History: {messages_str}
- Ground Truth Answer: {ground_truth}

### TASK: CONSTRUCT COT
Construct a correct Chain-of-Thought (CoT) that logically leads to the Ground Truth.
**Requirements for Concise CoT:**
1. Remove redundant explanations, meta-talk, and obvious narration.
2. Retain only the minimal logical justification needed to support the final answer.
3. Use professional, direct, and concise language.
"""