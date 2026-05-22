import sys
import re
import os
import json
from itertools import combinations

def parse_experiment_steps(text):
    # Regular expression to match experiment steps until encountering a single line containing only a right parenthesis ")"
    # Match format: variable_name = <action_name>(parameter_list)
    # Capture groups:
    #   1: output variable name (e.g., "multimer_cells")
    #   2: action name (e.g., "Incubate cells with MHC multimers")
    #   3: parameter list (e.g., "cells=washed_cells,\nmultimer_pool=\"tetramer pool (23 nM each)\",\n...")
    #   Condition: parameter list continues until a single line of ")" (whitespace allowed around it)
    step_pattern = r'(\w+)\s*=\s*<([^>]+)>\(\s*([\s\S]*?)(?=\n\s*\)\s*$)'
    # Regular expression to match each parameter line
    # Match format: key=value or key=value,
    # Capture groups:
    #   1: parameter key (e.g., "cells")
    #   2: parameter value (e.g., "washed_cells" or "\"tetramer pool (23 nM each)\"")
    #   (?:,)? : optionally match a trailing comma at the end of the line, ignore the comma
    param_pattern = r'^\s*(\w+)\s*=\s*(.*?)\s*(?:,)?\s*$'
    steps = []
    
    for match in re.finditer(step_pattern, text, re.MULTILINE):
        output_var = match.group(1).strip()  # Extract output variable name
        action_name = match.group(2).strip()  # Extract action name
        params = match.group(3).strip()      # Extract parameter list
        
        param_dict = {}
        # Split the parameter list by lines, ignoring empty lines and single-line ")"
        param_lines = [line.strip() for line in params.split('\n') if line.strip() and line.strip() != ')']
        for line in param_lines:
            param_match = re.match(param_pattern, line)
            if param_match:
                key = param_match.group(1)       # Extract parameter key
                value = param_match.group(2).strip()  # Extract parameter value
                # If the value starts and ends with double quotes, remove the quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                param_dict[key] = value
        
        # Build the step dictionary
        steps.append({
            "action": action_name,
            "input": param_dict,
            "output": output_var
        })
    
    return steps


def identify_variable_types(steps):
    """
    Identify raw variables and generated variables in the experimental steps.
    Raw variables: variables that never appear as outputs in any step.
    Generated variables: variables that appear as outputs of some step.
    
    Returns:
        original_vars (set): set of raw variables
        generated_vars (set): set of generated variables (function outputs)
        output_to_step_map (dict): mapping from output variable name to the index of its generating step (for reverse lookup)
    """
    generated_vars = set()
    all_input_vars = set()
    output_to_step_map = {}

    for idx, step in enumerate(steps):
        output_var = step["output"]
        generated_vars.add(output_var)
        output_to_step_map[output_var] = idx  # Store the step index
        
        for input_val in step["input"].values():
            # Simple check whether it is a variable (non-string literal, non-numeric)
            # If input_val is a string and does not start and end with quotes, and is not purely numeric, consider it a variable
            if isinstance(input_val, str) and \
               not (input_val.startswith('"') and input_val.endswith('"')) and \
               not is_numeric_literal(input_val):
                all_input_vars.add(input_val)

    # Raw variables are those input variables that are not in the set of output variables of any step
    original_vars = all_input_vars - generated_vars

    return original_vars, generated_vars, output_to_step_map


def normalize_action(action):
    return re.sub(r'\s+', ' ', action.strip().lower())


def is_numeric_literal(value):
    return isinstance(value, str) and (
        value.replace('.', '', 1).isdigit() or
        (value.startswith('-') and value[1:].replace('.', '', 1).isdigit())
    )


def align_action_steps(gt_steps, pred_steps):
    gt_actions = [normalize_action(step["action"]) for step in gt_steps]
    pred_actions = [normalize_action(step["action"]) for step in pred_steps]
    n = len(gt_actions)
    m = len(pred_actions)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if gt_actions[i] == pred_actions[j]:
                dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + 1)
            dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j + 1], dp[i + 1][j])

    aligned_pairs = []
    i = n
    j = m
    while i > 0 and j > 0:
        if gt_actions[i - 1] == pred_actions[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(aligned_pairs))


def score_aligned_parameters(gt_steps, pred_steps, aligned_pairs):
    _, generated_vars_gt, _ = identify_variable_types(gt_steps)
    _, generated_vars_pred, _ = identify_variable_types(pred_steps)
    var_map_pred2gt = {}
    total_score = 0.0

    for gt_idx, pred_idx in aligned_pairs:
        step_gt = gt_steps[gt_idx]
        step_pred = pred_steps[pred_idx]
        keys_gt = set(step_gt["input"].keys())
        keys_pred = set(step_pred["input"].keys())
        shared_keys = keys_gt & keys_pred

        if not keys_gt and not keys_pred:
            key_score = 1.0
        else:
            key_score = (2 * len(shared_keys)) / (len(keys_gt) + len(keys_pred))

        if not shared_keys:
            value_score = 1.0 if not keys_gt and not keys_pred else 0.0
        else:
            correct_values = 0
            for key in shared_keys:
                value_gt = step_gt["input"][key]
                value_pred = step_pred["input"][key]
                gt_is_generated = value_gt in generated_vars_gt
                pred_is_generated = value_pred in generated_vars_pred

                if gt_is_generated and pred_is_generated:
                    correct_values += int(var_map_pred2gt.get(value_pred) == value_gt)
                elif not gt_is_generated and not pred_is_generated:
                    correct_values += 1

            value_score = correct_values / len(shared_keys)

        total_score += key_score * value_score
        var_map_pred2gt[step_pred["output"]] = step_gt["output"]

    return total_score


def compare_exp_steps_soft(gt_steps, pred_steps):
    """
    Rule-based soft score for workflows with different numbers of steps.
    Actions are aligned by LCS. The sequence score is action recall times
    action precision, so missing and extra steps are both penalized.
    Parameter score uses the same recall/precision form over aligned steps.
    """
    aligned_pairs = align_action_steps(gt_steps, pred_steps)
    gt_step_count = len(gt_steps)
    pred_step_count = len(pred_steps)
    matched_action_count = len(aligned_pairs)
    length_gap = abs(len(gt_steps) - len(pred_steps))
    if gt_step_count == 0 and pred_step_count == 0:
        action_sequence_similarity = 1.0
        parameter_acc = 1.0
        action_recall = 1.0
        action_precision = 1.0
        parameter_recall = 1.0
        parameter_precision = 1.0
    elif gt_step_count == 0 or pred_step_count == 0:
        action_sequence_similarity = 0.0
        parameter_acc = 0.0
        action_recall = 0.0
        action_precision = 0.0
        parameter_recall = 0.0
        parameter_precision = 0.0
    else:
        action_recall = matched_action_count / gt_step_count
        action_precision = matched_action_count / pred_step_count
        action_sequence_similarity = action_recall * action_precision
        parameter_score = score_aligned_parameters(gt_steps, pred_steps, aligned_pairs)
        parameter_recall = parameter_score / gt_step_count
        parameter_precision = parameter_score / pred_step_count
        parameter_acc = parameter_recall * parameter_precision

    return {
        "action_sequence_similarity": action_sequence_similarity,
        "parameter_acc": parameter_acc,
        "matched_action_count": matched_action_count,
        "gt_step_count": gt_step_count,
        "pred_step_count": pred_step_count,
        "missing_action_count": gt_step_count - matched_action_count,
        "extra_action_count": pred_step_count - matched_action_count,
        "length_gap": length_gap,
        "action_recall": action_recall,
        "action_precision": action_precision,
        "parameter_recall": parameter_recall,
        "parameter_precision": parameter_precision,
    }


def compare_exp_steps(gt_steps, pred_steps):
    def kendall_tau_distance(seq1, seq2):
        if len(seq1) != len(seq2):
            return 0.0
        n = len(seq1)
        if n <= 1:
            return 1.0
        inversions = 0
        for i, j in combinations(range(n), 2):
            if (seq1[i] < seq1[j] and seq2[i] > seq2[j]) or (seq1[i] > seq1[j] and seq2[i] < seq2[j]):
                inversions += 1
        max_inversions = n * (n - 1) / 2
        return 1.0 - (inversions / max_inversions if max_inversions > 0 else 0.0)

    results = {
        "order_similarity": 0.0,
        "error_rate": 0.0,
        "details": []
    }
    
    actions_gt = [step["action"] for step in gt_steps]
    actions_pred = [step["action"] for step in pred_steps]
    
    results["order_similarity"] = kendall_tau_distance(actions_gt, actions_pred)
    
    # Identify variable types and build output mappings
    original_vars_gt, generated_vars_gt, output_to_step_map_gt = identify_variable_types(gt_steps)
    original_vars_pred, generated_vars_pred, output_to_step_map_pred = identify_variable_types(pred_steps)  # output_to_step_map_pred is only used to judge whether an input is a generated variable
    
    # Dictionary mapping variable names in pred_steps to corresponding variables in gt_steps
    var_map_pred2gt = {}
    
    error_count = 0
    min_len = min(len(gt_steps), len(pred_steps))
    
    for i in range(min_len):
        step_gt = gt_steps[i]
        step_pred = pred_steps[i]
        detail = {
            "step": i + 1,
            "action_gt": step_gt["action"],
            "action_pred": step_pred["action"],
            "status": "✅ success",
            "message": ""
        }
        
        # 1. Check whether the action names match
        if step_gt["action"] != step_pred["action"]:
            detail["status"] = "❌ error"
            detail["message"] += f"Action mismatch: expected '{step_gt['action']}', got '{step_pred['action']}'. "
            error_count += 1
            results["details"].append(detail)
            continue
            
        # 2. Check the set of parameter keys
        keys_gt = set(step_gt["input"].keys())
        keys_pred = set(step_pred["input"].keys())
        if keys_gt != keys_pred:
            detail["status"] = "❌ error"
            detail["message"] += f"Parameter keys mismatch: expected {keys_gt}, got {keys_pred}. "
            error_count += 1
            results["details"].append(detail)
            continue
        
        # 3. Check argument passing
        is_step_error = False  # Flag whether the current step has parameter errors
        for key in keys_gt:
            value_gt = step_gt["input"][key]
            value_pred = step_pred["input"][key]
            
            # Determine whether the parameter is a raw variable or a generated variable
            is_input_var_gt_generated = value_gt in generated_vars_gt
            is_input_var_pred_generated = value_pred in generated_vars_pred

            # Case 1: Both gt_steps and pred_steps inputs are generated variables (outputs from previous steps)
            if is_input_var_gt_generated and is_input_var_pred_generated:
                # Try mapping variables from pred_steps to the corresponding variables in gt_steps
                mapped_value_pred = var_map_pred2gt.get(value_pred)
                
                # If the variable from pred_steps successfully maps to the corresponding variable in gt_steps,
                # and the mapped value matches the expected value in gt_steps
                if mapped_value_pred == value_gt:
                    pass  # Match succeeds; continue
                else:
                    detail["status"] = "❌ error"
                    detail["message"] += f"Parameter '{key}' generated variable reference mismatch: expected from '{value_gt}', got from '{value_pred}' (mapped as '{mapped_value_pred}'). "
                    is_step_error = True
            # Case 2: Both inputs are raw variables (literals or inputs not defined as function outputs)
            elif not is_input_var_gt_generated and not is_input_var_pred_generated:
                # For raw variables, do not strictly require identical values; even if values differ, consider it correct
                pass 
            # Case 3: Type mismatch (one is a raw variable, the other is a generated variable)
            else:
                detail["status"] = "❌ error"
                detail["message"] += f"Parameter '{key}' type mismatch: expected {'generated variable' if is_input_var_gt_generated else 'raw variable'}, got {'generated variable' if is_input_var_pred_generated else 'raw variable'}. "
                is_step_error = True
        
        # If the current step has no parameter errors, update the variable mapping
        if not is_step_error:
            # Only when the action and parameters both match,
            # map the output variable in pred_steps to the output variable in gt_steps
            var_map_pred2gt[step_pred["output"]] = step_gt["output"]
        else:
            # If the step has errors, increment the error count
            error_count += 1
            
        results["details"].append(detail)
    
    # Handle the case where lengths are inconsistent
    if len(gt_steps) != len(pred_steps):
        error_count += abs(len(gt_steps) - len(pred_steps))
        if len(pred_steps) > len(gt_steps):
            for i in range(min_len, len(pred_steps)):
                results["details"].append({
                    "step": i + 1,
                    "action_gt": None,
                    "action_pred": pred_steps[i]["action"],
                    "status": "❌ error",
                    "message": "Extra step."
                })
        elif len(gt_steps) > len(pred_steps):
            for i in range(min_len, len(gt_steps)):
                results["details"].append({
                    "step": i + 1,
                    "action_gt": gt_steps[i]["action"],
                    "action_pred": None,
                    "status": "❌ error",
                    "message": "Missing step."
                })
    
    results["parameter_acc"] = 1 - (error_count/max(len(gt_steps), len(pred_steps)))
    
    return results

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

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'r', encoding='utf-8') as json_file:
    model_answer = json.load(json_file)

for ques_dict in model_answer:
    target_steps = ques_dict['answer']
    target_steps = parse_experiment_steps(target_steps)
    prediction_steps = ques_dict['model_answer']
    # prediction_steps = question['model_answer_after_llm_paser']
    prediction_steps = parse_experiment_steps(prediction_steps)

    result = compare_exp_steps(target_steps, prediction_steps)
    soft_result = compare_exp_steps_soft(target_steps, prediction_steps)
    # print("Order similarity:", result["order_similarity"])
    # print("Parameter accuracy:", result["parameter_acc"])
    # for detail in result["details"]:
    #     print(f"Step {detail['step']}: {detail['status']} - {detail['message']}")
    # print('---------------------------------')
    
    ques_dict['action_sequence_similarity'] = result['order_similarity']
    ques_dict['parameter_accuracy'] = result['parameter_acc']
    ques_dict['final_score'] = (ques_dict['action_sequence_similarity']+ques_dict['parameter_accuracy'])/2
    ques_dict['action_sequence_similarity_strict'] = ques_dict['action_sequence_similarity']
    ques_dict['parameter_accuracy_strict'] = ques_dict['parameter_accuracy']
    ques_dict['final_score_strict'] = ques_dict['final_score']
    ques_dict['soft_action_sequence_similarity'] = soft_result['action_sequence_similarity']
    ques_dict['soft_parameter_accuracy'] = soft_result['parameter_acc']
    ques_dict['soft_final_score'] = (
        ques_dict['soft_action_sequence_similarity'] + ques_dict['soft_parameter_accuracy']
    ) / 2
    ques_dict['gt_step_count'] = soft_result['gt_step_count']
    ques_dict['pred_step_count'] = soft_result['pred_step_count']
    ques_dict['soft_matched_action_count'] = soft_result['matched_action_count']
    ques_dict['soft_missing_action_count'] = soft_result['missing_action_count']
    ques_dict['soft_extra_action_count'] = soft_result['extra_action_count']
    ques_dict['soft_length_gap'] = soft_result['length_gap']
    ques_dict['soft_action_recall'] = soft_result['action_recall']
    ques_dict['soft_action_precision'] = soft_result['action_precision']
    ques_dict['soft_parameter_recall'] = soft_result['parameter_recall']
    ques_dict['soft_parameter_precision'] = soft_result['parameter_precision']

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(model_answer, json_file, ensure_ascii=False, indent=4)

print(model_name)
length_mismatch_count = sum(item['gt_step_count'] != item['pred_step_count'] for item in model_answer)
soft_nonzero_mismatch_count = sum(
    item['gt_step_count'] != item['pred_step_count'] and item['soft_action_sequence_similarity'] > 0
    for item in model_answer
)
print(f"Strict Action Sequence Similarity: {sum([item['action_sequence_similarity'] for item in model_answer])/len(model_answer)}")
print(f"Strict Parameter Accuracy: {sum([item['parameter_accuracy'] for item in model_answer])/len(model_answer)}")
print(f"Strict Final Score: {sum([item['final_score'] for item in model_answer])/len(model_answer)}")
print(f"Soft Action Sequence Similarity: {sum([item['soft_action_sequence_similarity'] for item in model_answer])/len(model_answer)}")
print(f"Soft Parameter Accuracy: {sum([item['soft_parameter_accuracy'] for item in model_answer])/len(model_answer)}")
print(f"Soft Final Score: {sum([item['soft_final_score'] for item in model_answer])/len(model_answer)}")
print(f"Length Mismatches: {length_mismatch_count}/{len(model_answer)}")
print(f"Length Mismatches with Non-zero Soft ASS: {soft_nonzero_mismatch_count}/{length_mismatch_count}")
