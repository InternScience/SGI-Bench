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

def eval_model_output(ques_dict):
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
{'\n'.join(ques_dict['steps'])}

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
    
    try:
        llm_judge = judge(prompt)
        start_index = llm_judge.find('[')
        end_index = llm_judge.rfind(']') + 1
        llm_judge = eval(repair_json(llm_judge[start_index:end_index]))
        correct_step_count = 0
        for step in llm_judge:
            if step["judge"] == "correct":
                correct_step_count += 1
        step_level_acc = correct_step_count / len(llm_judge)
    except:
        llm_judge = None

    ques_dict['exact_match'] = 1 if (ques_dict['answer'] == ques_dict['model_answer'] or ques_dict['answer'] == ques_dict['model_answer_after_llm_paser']) else 0
    ques_dict['llm_judge'] = llm_judge
    ques_dict['step_level_acc'] = step_level_acc
    return ques_dict


inp_list = [{'ques_dict': ques} for ques in model_answer]
out_list = multi_thread(inp_list, eval_model_output, 100)

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)


print(model_name)
print(f"Exact Match: {sum([item['exact_match'] for item in out_list])/len(out_list)}")
print(f"Step Level Acc: {sum([item['step_level_acc'] for item in out_list])/len(out_list)}")