import json
import os
import sys
sys.path.append('.')
from utils import LLM, multi_thread, load_sgi_dataset
import re
from typing import Dict, Any

dataset = load_sgi_dataset("InternScience/SGI-IdeaGeneration")
save_dir = './task_2_idea_generation/logs'
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

def parse_generated_idea(text: str) -> Dict[str, Any]:
    """Parse the generated research proposal text into a structured dictionary"""
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    json_block_match = re.search(json_block_pattern, text)
    if json_block_match:
        json_str = json_block_match.group(1).strip()
        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError:
            pass

    try:
        parsed_data = json.loads(text)
        return parsed_data
    except json.JSONDecodeError:
        pass

    result = {}
    
    idea_patterns = [
        r"[\"']?Idea[\"']?\s*:\s*[\"'](.*?)[\"']",
        r"1\.\s*Idea[:\s-]+(.*?)(?=\n\s*(?:2\.|Implementation))",
    ]
    for pattern in idea_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["Idea"] = match.group(1).strip()
            break

    steps_patterns = [
        r"[\"']?ImplementationSteps[\"']?\s*:\s*\{(.*?)\}",
        r"2\.\s*Implementation Steps[:\s-]+(.*?)(?=\n\s*(?:3\.|Implementation Order))",
    ]
    for pattern in steps_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            steps_text = match.group(1).strip()
            steps_dict = {}
            step_matches = re.findall(r"[\"'](\d+)[\"']\s*:\s*[\"'](.*?)[\"']", steps_text)
            for step_num, step_desc in step_matches:
                steps_dict[step_num] = step_desc.strip()
            if steps_dict:
                result["ImplementationSteps"] = steps_dict
                break

    order_patterns = [
        r"[\"']?ImplementationOrder[\"']?\s*:\s*\[(.*?)\]",
        r"3\.\s*Implementation Order[:\s-]+(.*?)(?=\n\s*(?:4\.|Dataset))",
    ]
    for pattern in order_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            order_text = match.group(1).strip()
            order_list = re.findall(r'["\']([^"\']+)["\']', order_text)
            if order_list:
                result["ImplementationOrder"] = order_list
                break

    dataset_patterns = [
        r"[\"']?Dataset[\"']?\s*:\s*[\"'](.*?)[\"'](?=\s*,\s*[\"'])",
        r"4\.\s*Dataset[:\s-]+(.*?)(?=\n\s*(?:5\.|Evaluation))",
    ]
    for pattern in dataset_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["Dataset"] = match.group(1).strip()
            break

    metrics_patterns = [
        r"[\"']?EvaluationMetrics[\"']?\s*:\s*\{(.*?)\}(?=\s*,\s*[\"'])",
        r"5\.\s*Evaluation Metrics[:\s-]+(.*?)(?=\n\s*(?:6\.|Expected))",
    ]
    for pattern in metrics_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            metrics_text = match.group(1).strip()
            metrics_dict = {}
            metric_matches = re.findall(r"[\"']([^\"']+)[\"']\s*:\s*[\"'](.*?)[\"']", metrics_text)
            for metric_name, metric_desc in metric_matches:
                metrics_dict[metric_name.strip()] = metric_desc.strip()
            if metrics_dict:
                result["EvaluationMetrics"] = metrics_dict
                break

    outcome_patterns = [
        r"[\"']?ExpectedOutcome[\"']?\s*:\s*[\"'](.*?)[\"']",
        r"6\.\s*Expected Outcome[:\s-]+(.*?)$",
    ]
    for pattern in outcome_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result["ExpectedOutcome"] = match.group(1).strip()
            break

    if not result:
        result["full_text"] = text

    return result

example = {
    "Idea": "We propose an adaptive optimization framework based on a dynamic feature interaction network. This framework captures feature correlations through a hierarchical attention mechanism and combines it with a data distribution-aware dynamic weight adjustment strategy to improve the model's adaptability to heterogeneous data while ensuring computational efficiency.",
    "ImplementationSteps": {
        "1": "Data preprocessing: missing value filling, outlier handling, feature normalization and type conversion, and building a basic feature set",
        "2": "Feature engineering: generating statistically derived features, time series features, and cross-features, and building a feature candidate pool",
        "3": "Model architecture design: building a basic network module, integrating a hierarchical attention mechanism with a dynamic interaction layer",
        "4": "Dynamic weight mechanism implementation: designing a data distribution-aware weight adjustment function and embedding it into the network's intermediate layers",
        "5": "Model training and tuning: adopting a phased training strategy, using grid search and early stopping to optimize hyperparameters",
        "6": "Performance Verification: Conduct comparative experiments on multiple datasets to analyze model performance differences in different scenarios."
    },
    "ImplementationOrder": ["1-2", "2-3", "3-4", "4-5", "1-5", "5-6"],
    "Dataset": "Contains three types of public datasets and one actual business data: 1) Public structured dataset (approximately 500,000 samples, 30+ features); 2) Text-numeric mixed dataset (approximately 200,000 samples, including text embedding features); 3) Time series sparse dataset (approximately 100,000 samples, spanning 1 year); 4) Real transaction data from an e-commerce platform (approximately 1 million samples, including user behavior and product attribute features)",
    "EvaluationMetrics": {
        "Prediction Accuracy": "AUC and F1-score are used for classification tasks; MAE and RMSE are used for regression tasks to evaluate the basic predictive ability of the model.",
        "Robustness": "Performance decay rate is calculated through data perturbation testing (adding noise and simulating feature loss) to measure model stability.",
        "Efficiency": "Record model training time, inference latency, and memory usage to evaluate computing resource consumption.",
        "Interpretability": "Use SHAP values and feature importance ranking to quantify the feature contribution to model decisions.",
        "Generalization": "Performance retention across datasets to evaluate the model's adaptability to unseen data."
    },
    "ExpectedOutcome": "The proposed framework outperforms existing mainstream methods in comprehensive performance (accuracy, robustness, and efficiency) across multiple datasets, particularly in scenarios with uneven data distribution and cross-scenario migration. It also enhances model interpretability through a dynamic feature interaction mechanism, providing effective support for practical business decision-making."
}

def get_answer(ques_dict: dict):
    try:
        prompt = ques_dict['question']+f"""\n\n### Example:
```json
{json.dumps(example, indent=4)}
```"""
        
        generated_idea = llm_model(prompt)
        generated_idea = str(generated_idea)
        
    except Exception as e:
        generated_idea = f"[Error][{str(e)}]"
        print(f"Error generating idea: {str(e)}")
    
    
    parsed_data = parse_generated_idea(generated_idea)
    
    ques_dict['generated_idea_text'] = generated_idea
    ques_dict['generated_data'] = parsed_data
    
    return ques_dict

inp_list = [{"ques_dict": q} for q in dataset['test'] if q['discipline'] in discipline_list]

out_list = multi_thread(inp_list, get_answer)

os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json")
with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)

print(f"Results saved to {output_path}")
print(f"Total processed: {len(out_list)}")
