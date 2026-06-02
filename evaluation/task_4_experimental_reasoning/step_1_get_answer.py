import json
import os
import sys
sys.path.append('.')
from utils import VLM, multi_thread, load_sgi_dataset

dataset = load_sgi_dataset("InternScience/SGI-Reasoning")
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

vlm_model = VLM(model_name)


CONTEXT = """
Please solve the following multiple-choice question step-by-step. Each question is provided with several options labeled A, B, C, D, E, etc. Carefully analyze the question and each option, reason step-by-step, then select the single most correct option.

Your final output **must** include both **the reasoning** and **the final answer**. The final answer must meet two core requirements:
1. It consists solely of the corresponding letter of the correct option (e.g., A, B, C, D, E, etc.);
2. This letter is enclosed in the \\boxed{} format. Example: \\boxed{A}
    """.strip()


def get_answer(ques_dict: dict):
    prompt = CONTEXT + "\n\n"
    prompt += "Question:\n" + ques_dict['question'] + "\n\n"
    prompt += "Options:\n"
    for i, option in enumerate(ques_dict['options']):
        option_label = chr(ord('A') + i)
        prompt += f"{option_label}. {option}\n"

    try:
        answer = vlm_model(ques_dict["images"], prompt)
        answer = str(answer)
    except Exception as e:
        answer = f"[Error][{str(e)}]"
        print(str(e))

    ques_dict['model_answer'] = answer

    return ques_dict

inp_list = [{"ques_dict": q} for q in dataset['test'] if q['discipline'] in discipline_list]
out_list = multi_thread(inp_list, get_answer)

os.makedirs(save_dir, exist_ok=True)
for idx in range(len(out_list)):
    # unserializable
    del out_list[idx]['images']
    del out_list[idx]['step_images']
with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)
