import json
import os
import re
import sys

from datasets import load_dataset
sys.path.append('.')
from utils import VLM, multi_thread

dataset = load_dataset("InternScience/SGI-Reasoning")
save_dir = './task_4_experimental_reasoning/logs'
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

image_mapping = {q["idx"]: {"images": q["images"], "step_images": q["images"]} for q in dataset['test']}
for item in model_answer:
    idx = item["idx"]
    item['images'] = image_mapping[idx]['images']
    item['step_images'] = image_mapping[idx]['step_images']

judge = VLM('o4-mini')


def extract_answer_from_response(response):
    match = re.search(r"\\boxed\{([A-Z])\}", response)
    if match:
        return match.group(1)
    else:
        return None


def mm_reasoning_is_correct(pred, gold):
    try:
        ans = extract_answer_from_response(pred).strip()
    except:
        return False
    return ans.lower() == gold.lower()


CONTEXT = """
Please solve the following multiple-choice question step-by-step. Each question is provided with several options labeled A, B, C, D, E, etc. Carefully analyze the question and each option, reason step-by-step, then select the single most correct option.

Your final output **must** include both **the reasoning** and **the final answer**. The final answer must meet two core requirements:
1. It consists solely of the corresponding letter of the correct option (e.g., A, B, C, D, E, etc.);
2. This letter is enclosed in the \\boxed{} format. Example: \\boxed{A}
    """.strip()


def eval_model_output(ques_dict):
    prompt = CONTEXT + "\n\n"
    prompt += "Question:\n" + ques_dict['question'] + "\n\n"
    prompt += "Options:\n"
    for i, option in enumerate(ques_dict['options']):
        option_label = chr(ord('A') + i)
        prompt += f"{option_label}. {option}\n"

    reference_steps = ques_dict["steps"]
    reference_steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(reference_steps)])

    prompt = f"""You are a strict evaluator assessing the **validity of the model prediction's reasoning process**. You must score this reasoning validity on a scale from 0 to 10, where 0 means the reasoning is completely invalid and 10 means the reasoning is fully rigorous.
# Input
Question:
```
{prompt}
```
Reference Reasoning:
```
{reference_steps}
```
Model Prediction:
```
{ques_dict['model_answer']}
```
# Evaluation Rules
1. First, identify the **complete reasoning process** from the model prediction (ignore only the final answer if it is not accompanied by reasoning).
2. Evaluate reasoning validity against two core criteria:
   - **Logical Coherence**: Check if the reasoning steps are sequential, self-consistent, and free of contradictions (e.g., no conflicting premises or illogical deductions).
   - **Alignment with Reference Reasoning**: Check if the reasoning direction, key premises, and deduction logic match the reference reasoning (partial alignment counts for partial credit).
3. Deduct points for:
   - Irrelevant content (reasoning that does not address the question or key conditions).
   - Missing key reasoning steps (even if the final answer is correct).
   - Flawed logic (e.g., circular reasoning, false premises leading to conclusions).
4. Do not prioritize the correctness of the **final answer**—a correct answer with invalid reasoning still scores low, while an incorrect answer with partially valid reasoning may score higher.
# Scoring Guide
- **10**: Reasoning is fully rigorous, logically coherent (no contradictions), and perfectly aligned with the reference reasoning (all key steps and logic match).
- **7-9**: Reasoning is mostly coherent, with minor logical gaps or partial misalignment with the reference reasoning (no major contradictions).
- **4-6**: Reasoning has obvious logical flaws (e.g., one missing key step, minor contradictions) or limited alignment with the reference reasoning (only some core logic matches).
- **1-3**: Reasoning is barely valid, with severe logical flaws (e.g., multiple contradictions) or almost no alignment with the reference reasoning (only tangentially related to the question).
- **0**: Reasoning is completely invalid, contradictory (self-conflicting logic), or irrelevant (no connection to the question or key conditions).
# Strict Output format example
6"""

    try:
        llm_judge = judge(ques_dict["images"], prompt).strip()
        pattern = r"(\d+)"
        match = re.search(pattern, llm_judge)
        rv_score = float(match.group(1)) if match else 0.0
    except:
        rv_score = 0.0

    mcc_score = mm_reasoning_is_correct(ques_dict['model_answer'], chr(ord('A') + int(ques_dict['answer'])))

    ques_dict['MCA'] = 1 if mcc_score else 0
    ques_dict['RV'] = rv_score / 10.0
    return ques_dict


inp_list = [{'ques_dict': ques} for ques in model_answer]
out_list = multi_thread(inp_list, eval_model_output, 100)

for idx in range(len(out_list)):
    # unserializable
    del out_list[idx]['images']
    del out_list[idx]['step_images']
with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)

print(model_name)
print(f"MCA: {sum([item['MCA'] for item in out_list])/len(out_list)}")
print(f"RV: {sum([item['RV'] for item in out_list])/len(out_list)}")