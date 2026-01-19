import json
import os
import sys
from json_repair import repair_json
sys.path.append('.')
from utils import LLM, multi_thread


save_dir = './task_3_dry_experiment/logs'
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
    for unit_test_idx in range(5):
        unit_test_dict = ques_dict['unit_test'][unit_test_idx]
        correct_output = ques_dict[f"unit_test_{unit_test_idx}_output"]
        unit_test_dict['exact_match'] = 1 if (unit_test_dict['model_code_output'] == correct_output) else 0

        if unit_test_dict["exact_match"]:
            unit_test_dict["llm_judge"] = {"judgment": "correct", "reason": "Exact match."}
            unit_test_dict['pass'] = 1
            ques_dict['unit_test'][unit_test_idx] = unit_test_dict
            continue
        
        if unit_test_dict["model_error"].startswith("[WRONG]") or unit_test_dict["model_returncode"] != 0:
            unit_test_dict["llm_judge"] = {"judgment": "incorrect", "reason": "There are problems running the completed code."}
            unit_test_dict['pass'] = 0
            ques_dict['unit_test'][unit_test_idx] = unit_test_dict
            continue

        prompt = f"""
You are an expert in evaluating model output accuracy. Your task is to precisely determine whether the model output matches the reference output and provide a brief explanation.

## Instructions
1. Check all numerical values and ensure strict accuracy—every digit must match exactly. Any inconsistency should be considered incorrect.
2. For training-related loss values or metrics, if the difference between model output and reference output loss or metric values is greater than 2%, consider it incorrect.
3. The output should be a dictionary without any other text in the following format:
example = {{
    "judgment": "Placeholder, use 'correct' if outputs match, 'incorrect' otherwise",
    "reason": "Brief explanation placeholder"
}}

## Reference Output
{correct_output}

## Model Output
{unit_test_dict["model_code_output"]}
"""
        
        try:
            llm_judge = judge(prompt)
            start_index = llm_judge.find('{')
            end_index = llm_judge.rfind('}') + 1
            llm_judge = eval(repair_json(llm_judge[start_index:end_index]))
        except:
            llm_judge = None

        unit_test_dict['llm_judge'] = llm_judge
        unit_test_dict['pass'] = 1 if unit_test_dict['llm_judge']['judgment'] == 'correct' else 0
        ques_dict['unit_test'][unit_test_idx] = unit_test_dict
    
    ques_dict['pass_nums'] = sum([unit_test_dict['pass'] for unit_test_dict in ques_dict['unit_test']])
    ques_dict['PassAll@5'] = 1 if ques_dict['pass_nums'] == 5 else 0
    ques_dict['PassAll@3'] = 1 if ques_dict['pass_nums'] >= 3 else 0
    ques_dict['PassAll@1'] = 1 if ques_dict['pass_nums'] >= 1 else 0
    ques_dict['AET'] = [unit_test_dict['model_runtime'] for unit_test_dict in ques_dict['unit_test'] if unit_test_dict['model_runtime'] > 0]
    ques_dict['AET'] = sum(ques_dict['AET'])/len(ques_dict['AET']) if len(ques_dict['AET']) > 0 else -1
    ques_dict['SER'] = sum([1 if unit_test_dict['model_returncode']==0 else 0 for unit_test_dict in ques_dict['unit_test']]) / 5
    return ques_dict


inp_list = [{'ques_dict': ques} for ques in model_answer]
out_list = multi_thread(inp_list, eval_model_output, 100)

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)

print(model_name)
print(f"PassAll@5: {sum([item['PassAll@5'] for item in out_list])/len(out_list)}")
print(f"PassAll@3: {sum([item['PassAll@3']  for item in out_list])/len(out_list)}")
print(f"PassAll@1: {sum([item['PassAll@1'] for item in out_list])/len(out_list)}")
print(f"AET: {sum([item['AET'] for item in out_list if item['AET'] > 0])/len([item['AET'] for item in out_list if item['AET'] > 0])}")
print(f"SER: {sum([item['SER'] for item in out_list])/len(out_list)}")