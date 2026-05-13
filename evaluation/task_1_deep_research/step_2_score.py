import json
import os
import sys
sys.path.append('.')
from utils import LLM, multi_thread
from json_repair import repair_json


save_dir = './task_1_deep_research/logs'
model_name = 'gpt-4.1'
discipline = "['all']"
discipline_list = ['astronomy', 'chemistry', 'earth', 'energy', 'information', 'life', 'material', 'mathematics', 'neuroscience', 'physics']
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    sys.argv = sys.argv[1:]
if len(sys.argv) > 1:
    discipline = sys.argv[1]
    discipline_list = eval(discipline)
    sys.argv = sys.argv[1:]
print(f'Evaluating {model_name} in {discipline}')

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'r', encoding='utf-8') as json_file:
    model_answer = json.load(json_file)

judge = LLM('o4-mini')

def parse_repaired_json_object(text):
    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    if start_index == -1 or end_index <= start_index:
        raise ValueError("No JSON object found in judge response.")
    return json.loads(repair_json(text[start_index:end_index]))


def answer_llm_judge(ques_dict, exact_match):
    if exact_match:
        return 1, "Exact match."

    prompt = f"""You are a scientific answer evaluator. Compare the agent's answer to the reference answer for the following question.

Question: {ques_dict['question']}

Reference Answer: {ques_dict['answer']}

Agent's Answer: {ques_dict['model_answer']}

Parser-normalized Agent's Answer: {ques_dict.get('model_answer_after_llm_paser', '')}

Evaluate whether the agent's answer is essentially correct. Consider:
- For numerical answers: accept if within 5% relative error after accounting for obvious units, percentage signs, or formatting differences when the context supports the same meaning
- For text answers: accept if the meaning is equivalent
- Partial credit is NOT given - answer is either correct (1) or incorrect (0)

Respond with a JSON object: {{"judge": 0 or 1, "reason": "brief explanation"}}"""

    try:
        response = judge(prompt)
        result = parse_repaired_json_object(response)
        return int(result.get("judge", 0)), result.get("reason", "")
    except Exception as e:
        return 0, f"Judge error: {e}"


def eval_model_output(ques_dict):
    reference_steps = '\n'.join(ques_dict['steps'])
    prompt = f"""
You are an expert in systematically validating and evaluating LLM-generated solutions. Your task is to rigorously analyze the correctness of a provided solution by comparing it step-by-step against the reference solution, and output **only** a structured verification list—with no additional text.

## Instructions  
1. Break down the given LLM solution into individual steps and evaluate each one against the corresponding reference solution steps.  
2. For each step, include the following three components:  
   - **solution_step**: The specific part of the LLM solution being evaluated.  
   - **reason**: A clear, critical explanation of whether the step contains errors, omissions, or deviations from the reference approach. Be stringent in your assessment.  
   - **judge**: Your verdict: either `"correct"` or `"incorrect"`.  
3. If the final LLM answer is incorrect, you must identify at least one step in your analysis as incorrect.  
4. Justify your judgments rigorously, pointing out even minor inaccuracies or logical flaws.  
5. Do not attempt to answer the original question—your role is strictly to evaluate.  
6. Output **only** a list of dictionaries in the exact format provided below. Do not include any other text or comments.

## Question  
{ques_dict['question']}

## Reference Solution Steps  
{reference_steps}

## Reference Answer  
{ques_dict['answer']}

## LLM Solution Steps
{ques_dict['model_answer_with_thinking']}

## LLM Answer
{ques_dict['model_answer']}

## Output Example  
[  
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},  
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},
]
"""

    exact_match = 1 if (ques_dict['answer'] == ques_dict['model_answer'] or ques_dict['answer'] == ques_dict.get('model_answer_after_llm_paser', '')) else 0
    llm_judge, llm_judge_reason = answer_llm_judge(ques_dict, exact_match)
    step_llm_judge = None
    step_level_acc = 0.0

    try:
        step_llm_judge = judge(prompt)
        start_index = step_llm_judge.find('[')
        end_index = step_llm_judge.rfind(']') + 1
        step_llm_judge = eval(repair_json(step_llm_judge[start_index:end_index]))
        correct_step_count = 0
        for step in step_llm_judge:
            if step["judge"] == "correct":
                correct_step_count += 1
        step_level_acc = correct_step_count / len(step_llm_judge) if len(step_llm_judge) > 0 else 0.0
    except Exception as e:
        step_llm_judge = None

    ques_dict['exact_match'] = exact_match
    ques_dict['llm_judge'] = llm_judge
    ques_dict['llm_judge_reason'] = llm_judge_reason
    ques_dict['step_llm_judge'] = step_llm_judge
    ques_dict['step_level_acc'] = step_level_acc
    return ques_dict


inp_list = [{'ques_dict': ques} for ques in model_answer]
out_list = multi_thread(inp_list, eval_model_output, 100)

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)


print(model_name)
print(f"Exact Match: {sum([item['exact_match'] for item in out_list])/len(out_list)}")
print(f"LLM Judge: {sum([item['llm_judge'] for item in out_list])/len(out_list)}")
print(f"Step Level Acc: {sum([item['step_level_acc'] for item in out_list])/len(out_list)}")
