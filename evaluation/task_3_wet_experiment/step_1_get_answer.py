import json
import os
import sys
sys.path.append('.')
from utils import LLM, AnswerPaser, multi_thread, extract_final_answer
from datasets import load_dataset

dataset = load_dataset("InternScience/SGI-WetExperiment")
save_dir = './task_3_wet_experiment/logs'
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

llm_model = LLM(model_name)
answer_paser = AnswerPaser()
output_requirements = """
The final answer should be enclosed by <answer> and </answer>.

Example:
<answer>
dataset = <Load dataset>(
    source="imagenet"
)

model_init = <Initialize model>(
    model_type="CNN"
)

model_trained = <Train model>(
    model=model_init,
    data=dataset
)

metrics = <Calculate metrics>(
    model=model_trained,
    data=dataset
)
</answer>
"""

answer_example = """
dataset = <Load dataset>(
    source="imagenet"
)

model_init = <Initialize model>(
    model_type="CNN"
)

model_trained = <Train model>(
    model=model_init,
    data=dataset
)

metrics = <Calculate metrics>(
    model=model_trained,
    data=dataset
)
"""


def get_answer(ques_dict: dict):
    question = ques_dict['question']
    
    try:
        answer_with_thinking = llm_model(question+output_requirements)
        answer_with_thinking = str(answer_with_thinking)
    except Exception as e:
        answer_with_thinking = f"[Error][{str(e)}]"
        print(str(e))
    
    if answer_with_thinking.lower()=='none' or len(answer_with_thinking)==0:
        answer_with_thinking = "None"
        answer = "None"
        answer_after_llm_paser = "None"
    elif answer_with_thinking.startswith("[Error]"):
        answer = "None"
        answer_after_llm_paser = "None"
    else:
        answer = extract_final_answer(answer_with_thinking)
        if answer is None:
            answer = answer_with_thinking
        
        try:
            answer_after_llm_paser = answer_paser(answer, answer_example)
            answer_after_llm_paser = str(answer_after_llm_paser)
        except:
            answer_after_llm_paser = answer

    ques_dict['model_answer_with_thinking'] = answer_with_thinking
    ques_dict['model_answer'] = answer
    ques_dict['model_answer_after_llm_paser'] = answer_after_llm_paser

    return ques_dict

inp_list = [{"ques_dict": q} for q in dataset['test'] if q['discipline'] in discipline_list]
out_list = multi_thread(inp_list, get_answer)

os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)