import os
import json
import re
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import sys
import networkx as nx
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import ast
import time
sys.path.append('.')
from utils import LLM, multi_thread
from utils import format_idea_data, get_context_from_data, get_evaluation_prompt_modified, parse_evaluation_result, flip_evaluation_result

dataset = load_dataset("InternScience/SGI-IdeaGeneration")
save_dir = './task_2_idea_generation/logs'
model_name = "gpt-4.1"
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

MAX_RETRIES=5
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

JUDGE_MODELS = ["gpt-5.1-2025-11-13", "gemini-3-pro-preview", "anthropic/claude-sonnet-4.5"] # specify three judge models
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")
    embedding_model = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    return np.dot(a_normalized, b_normalized.T)

def edge_jaccard(G1, G2):
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    if not edges1 and not edges2:
        return 1.0
    return len(edges1 & edges2) / len(edges1 | edges2)


def node_text_similarity(G1, G2):
    texts1 = [G1.nodes[n]['text'] for n in G1.nodes()]
    texts2 = [G2.nodes[n]['text'] for n in G2.nodes()]
    
    if not texts1 or not texts2:
        logging.warning("node_text_similarity: One of the graphs has no node texts.")
        return 0.0
    
    try:
        combined_text1 = ' '.join(texts1)
        combined_text2 = ' '.join(texts2)
        
        if len(combined_text1.strip()) < 3 or len(combined_text2.strip()) < 3:
            logging.warning("node_text_similarity: One of the texts is too short to compare.")
            return 0.0
        
        words1 = set(combined_text1.lower().split())
        words2 = set(combined_text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        return jaccard_sim
    except Exception as e:
        return 0.0


def graph_similarity(dict1, dict2, alpha=0.5):
    if not all(k in dict1 for k in ["ImplementationSteps", "ImplementationOrder"]) or \
       not all(k in dict2 for k in ["ImplementationSteps", "ImplementationOrder"]):
        logging.warning("graph_similarity: One of the graphs is missing necessary keys.")
        return 0.0
    
    if not dict1["ImplementationSteps"] or not dict1["ImplementationOrder"] or \
       not dict2["ImplementationSteps"] or not dict2["ImplementationOrder"]:
        logging.warning("graph_similarity: One of the graphs is missing necessary keys.")
        return 0.0
    
    try:
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()
        
        for k, v in dict1["ImplementationSteps"].items():
            G1.add_node(str(k), text=v)
        for k, v in dict2["ImplementationSteps"].items():
            G2.add_node(str(k), text=v)
        
        if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
            return 0.0
        
        def process_order_items(order_list, graph, step_keys):
            edges_added = False
            if all(o.isdigit() for o in order_list):
                nodes = sorted([o for o in order_list if o in step_keys])
                for i in range(len(nodes) - 1):
                    graph.add_edge(nodes[i], nodes[i+1])
                    edges_added = True
            else:
                for o in order_list:
                    if "-" in o:
                        try:
                            src, dst = o.split("-")
                            if src in step_keys and dst in step_keys:
                                graph.add_edge(src, dst)
                                edges_added = True
                        except Exception as e:
                            pass
            return edges_added
        
        step_keys1 = [str(k) for k in dict1["ImplementationSteps"].keys()]
        step_keys2 = [str(k) for k in dict2["ImplementationSteps"].keys()]
        
        edges_added_G1 = process_order_items(dict1["ImplementationOrder"], G1, step_keys1)
        edges_added_G2 = process_order_items(dict2["ImplementationOrder"], G2, step_keys2)
        
        if not edges_added_G1:
            nodes1 = sorted([n for n in G1.nodes()])
            for i in range(len(nodes1) - 1):
                G1.add_edge(nodes1[i], nodes1[i+1])
                edges_added_G1 = True
        
        if not edges_added_G2:
            nodes2 = sorted([n for n in G2.nodes()])
            for i in range(len(nodes2) - 1):
                G2.add_edge(nodes2[i], nodes2[i+1])
                edges_added_G2 = True
        
        if not edges_added_G1 or not edges_added_G2:
            logging.warning("graph_similarity: One of the graphs has no edges, only node text similarity will be computed.")
            return node_text_similarity(G1, G2)
        
        edge_sim = edge_jaccard(G1, G2)
        text_sim = node_text_similarity(G1, G2)
        
        return alpha * edge_sim + (1 - alpha) * text_sim
    
    except Exception as e:
        return 0.0


def calculate_semantic_repetition(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?。！？]', text) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 0.0
    
    try:
        if embedding_model is None:
            logging.warning("embedding_model is not available, cannot compute semantic repetition")
            return 0.0
        
        sentence_embeddings = embedding_model.encode(sentences)
        similarity_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
        
        upper_triangle = []
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                upper_triangle.append(similarity_matrix[i][j])
        
        if not upper_triangle:
            return 0.0
        
        avg_similarity = np.mean(upper_triangle)
        penalty = max(0, (avg_similarity - 0.2) * 10)
        
        return min(penalty, 10.0)
    
    except Exception as e:
        logging.error(f"calculate_semantic_repetition error: {e}")
        return 0.0


def get_vote_from_model(model, original_idea_data, generated_idea_data, context=None, swap_positions=False):
    original_idea_text = format_idea_data(original_idea_data)
    generated_idea_text = format_idea_data(generated_idea_data)
    
    # determine positions for evaluation
    if swap_positions:
        # swap positions: generated idea as A, original idea as B
        prompt = get_evaluation_prompt_modified(generated_idea_text, original_idea_text, context)
        positions_swapped = True
    else:
        # default positions: original idea as A, generated idea as B
        prompt = get_evaluation_prompt_modified(original_idea_text, generated_idea_text, context)
        positions_swapped = False
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            llm_model=LLM(model)
            response = llm_model(
                prompt,
                temperature=0.1
            )
            
            if response is None:
                retry_count += 1
                logging.warning(f"model {model} API call failed, retry {retry_count}")
                time.sleep(1)  
                continue
            
            evaluation_result = parse_evaluation_result(response)
            if evaluation_result is None:
                retry_count += 1
                logging.warning(f"model {model} evaluation result parse error, retry {retry_count}")
                time.sleep(1) 
                continue
            
            if positions_swapped:
                evaluation_result = flip_evaluation_result(evaluation_result)
                
            return evaluation_result
        
        except Exception as e:
            retry_count += 1
            logging.error(f"model {model} evaluation error: {e}, try {retry_count}")
            time.sleep(1)   

    logging.warning(f"model {model} evaluation failed after {MAX_RETRIES} retries")
    return None

def compare_ideas_with_voting(original_idea_data, generated_idea_data, context=None, judge_models=JUDGE_MODELS):
    dimensions = ["effectiveness", "novelty", "detailedness", "feasibility", "overall"]
    
    vote_counts = {
        dim: {"original": 0, "generated": 0} for dim in dimensions
    }

    all_evaluations = []
    
    for model in judge_models:
        for swap in [False, True]:  # each model votes twice, once with normal positions, once with swapped positions
            evaluation = get_vote_from_model(
                model=model,
                original_idea_data=original_idea_data,
                generated_idea_data=generated_idea_data,
                context=context,
                swap_positions=swap
            )
            
            if evaluation:
                vote_detail = {
                    "model": model,
                    "positions_swapped": swap,
                    "results": {}
                }
                
                for dim in dimensions:
                    dim_result = evaluation.get(dim, {})
                    judgment = dim_result.get("judgment", "")
                    reason = dim_result.get("reason", "No reason provided")
                    
                    if judgment == "win_A":
                        vote_counts[dim]["original"] += 1
                        result = "original_wins"
                    elif judgment == "win_B":
                        vote_counts[dim]["generated"] += 1
                        result = "generated_wins"
                    else:
                        logging.warning(f"error: {judgment}")
                        continue
                        
                    vote_detail["results"][dim] = {
                        "result": result,
                        "reason": reason
                    }
                
                all_evaluations.append(vote_detail)
            else:
                logging.error(f"model {model} evaluation failed, could not get votes")
    
    final_results = {}
    for dim in dimensions:
        original_votes = vote_counts[dim]["original"]
        generated_votes = vote_counts[dim]["generated"]
        
        lose_gate=2
        if dim=="novelty":
            win_gate=4
        else:
            win_gate=3
            
        if generated_votes> win_gate:
            result = "win"
            reason = f"Generated idea received {generated_votes} votes, Original idea received {original_votes} votes."
        elif generated_votes<= lose_gate:
            result = "lose"
            reason = f"Original idea received {original_votes} votes, Generated idea received {generated_votes} votes."
        else :
            result = "tie"
            reason = f"Generated idea received {generated_votes} votes, Original idea received {original_votes} votes."
            
        final_results[dim] = {
            "res": result,
            "reason": reason,
            "vote_detail": {
                "original_votes": original_votes,
                "generated_votes": generated_votes
            }
        }
    
    return {
        "final_results": final_results,
        "all_evaluations": all_evaluations
    }
    

class ImprovedIdeaEvaluator:
    def __init__(self, idea_dict: dict):
        self.idea_dict = idea_dict
        self.original_data = {k: v for k, v in idea_dict.items() if k not in ["generated_idea_text", "generated_data"]}
        self.original_data["Idea"] = self.original_data.get("core_idea", "")
        self.original_data["RelatedWork"] = ast.literal_eval(self.original_data.get("related_work", {}))
        self.original_data["ExistingSolutions"] = ast.literal_eval(self.original_data.get("existing_solutions", {}))
        self.original_data["ImplementationSteps"] = ast.literal_eval(self.original_data.get("implementation_steps", "{}"))
        self.original_data["ImplementationOrder"] = list(self.original_data.get("implementation_order", "[]"))
        self.original_data["EvaluationMetrics"] = ast.literal_eval(self.original_data.get("evaluation_metrics", "{}"))
        self.original_data["Dataset"] = self.original_data.get("data", "")
        self.original_data["ExpectedOutcome"] = self.original_data.get("expected_outcome", "")
        
        self.generated_data = idea_dict["generated_data"]
        self.idea = self.generated_data.get("Idea", "")
        self.generated_data["Idea"]=self.idea
        self.implementation_steps = self.generated_data.get("ImplementationSteps", {})
        self.implementation_order = self.generated_data.get("ImplementationOrder", {})
        self.dataset = self.generated_data.get("Dataset", "")
        self.generated_data["Dataset"] = self.dataset
        self.evaluation_metrics = self.generated_data.get("EvaluationMetrics", "")
        self.expected_outcome = self.generated_data.get("ExpectedOutcome", "")
        
        self.raw_scores = {
            "novelty_similarity": 0.0,
            "cutting_edge": 0.0,
            "effectiveness_objective": 0.0,
            "feasibility_objective": 0.0,
            "completeness": 0.0,
            "length_penalty": 0.0,
            "repetition_penalty": 0.0
        }
        
        self.scores = {
            "novelty_objective": 0.0,
            "feasibility_objective": 0.0,
            "detailedness_objective": 0.0,
            "effectiveness_objective": 0.0,
            "novelty": "", 
            "effectiveness": "",
            "detailedness": "",
            "feasibility": "",
        }
        self.details = {}
    
    
    def evaluate_novelty_objective(self) -> None:
        try:
            text_to_compare = self.idea
            related_work = self.original_data.get("RelatedWork", {})
            existing_methods = self.original_data.get("ExistingSolutions", {})
            
            all_existing_text = []
            all_existing_text.extend(related_work.values())
            all_existing_text.extend(existing_methods.values())
            
            if all_existing_text and embedding_model is not None:
                idea_embedding = embedding_model.encode([text_to_compare])
                similarities = []
                for existing_text in all_existing_text:
                    existing_embedding = embedding_model.encode([existing_text])
                    similarity = cosine_similarity(
                        idea_embedding.reshape(1, -1),
                        existing_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                
                novelty_similarity_score = (1 - avg_similarity) * 10
                novelty_similarity_score = max(0, min(10, novelty_similarity_score))
                
            else:
                novelty_similarity_score = 0.0
            
            self.raw_scores["novelty_similarity"] = novelty_similarity_score
            
            ref_related_work=self.original_data.get("related_work_test", "")
            
            idea_embedding = embedding_model.encode([self.idea]) 
        
            similarities = []
            ref_related_work=ast.literal_eval(ref_related_work)
            
            for key, value in ref_related_work.items():
                snippet_data = f"{key}: {value}"
                snippet_embedding = embedding_model.encode([snippet_data])
                similarity = cosine_similarity(
                    idea_embedding.reshape(1, -1), 
                    snippet_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            cutting_edge_score = (1 - avg_similarity) * 10
            cutting_edge_score = max(0, min(10, cutting_edge_score)) 
            
            self.raw_scores["cutting_edge"] = cutting_edge_score
            
            
        except Exception as e:
            logging.error(f"Error in novelty evaluation: {e}")
            self.raw_scores["novelty_similarity"] = 0.0
            self.raw_scores["cutting_edge"] = 0.0
            self.details["novelty_similarity"] = f"error: {str(e)}"
            self.details["cutting_edge"] = f"error: {str(e)}"
    
    def evaluate_effectiveness_objective(self) -> None:
        try:
            original_terms = self.original_data.get("keywords", [])
            if embedding_model is None:
                self.scores["effectiveness_objective"] = 0.0
                self.details["effectiveness_objective"] = "embedding_model is not available"
                return
            
            terms_text = ", ".join([str(term) for term in original_terms])
            idea_text = self.idea 
            
            try:
                embeddings = embedding_model.encode([terms_text, idea_text], normalize_embeddings=True)
                similarity = np.dot(embeddings[0], embeddings[1])
                prof_score = similarity * 10
                self.scores["effectiveness_objective"] = max(0, min(10, prof_score))
            except Exception as e:
                logging.error(f"Error computing embedding similarity: {e}")
                matched_terms = []
                generated_text_lower = idea_text.lower() if isinstance(idea_text, str) else ""
                for term in original_terms:
                    term_str = str(term).lower()
                    if term_str in generated_text_lower:
                        matched_terms.append(term)
                hit_rate = len(matched_terms) / len(original_terms) if original_terms else 0
                self.scores["effectiveness_objective"] = hit_rate * 10
                similarity = hit_rate
            

        except Exception as e:
            logging.error(f"Error in effectiveness_objective evaluation: {e}")
            self.scores["effectiveness_objective"] = 0.0
    
    def evaluate_completeness(self) -> None:
        required_sections = [
            "Idea",
            "ImplementationSteps",
            "ImplementationOrder",
            "EvaluationMetrics",
            "Dataset",
            "ExpectedOutcome"
        ]
        
        section_found = {
            "Idea": self.idea is not None,
            "ImplementationSteps": self.implementation_steps is not None,
            "ImplementationOrder": self.implementation_order is not None,
            "EvaluationMetrics": self.evaluation_metrics is not None,
            "Data": self.dataset is not None,
            "ExpectedOutcome": self.expected_outcome is not None
        }
        
        total_sections = len(required_sections)
        completed_sections = sum(section_found.values())
        
        self.raw_scores["completeness"] = (completed_sections / total_sections) * 10
        
        self.details["completeness"] = {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "completion_rate": completed_sections / total_sections,
        }
        
        missing_sections = [section for section, found in section_found.items() if not found]
        if missing_sections:
            logging.warning(f"Missing required sections: {', '.join(missing_sections)}")

    
    def evaluate_feasibility_objective(self) -> None:
        try:
            generated_implementation = {
                "ImplementationSteps": self.implementation_steps,
                "ImplementationOrder": self.implementation_order
            }

            original_implementation = {
                "ImplementationSteps": self.original_data["ImplementationSteps"],
                "ImplementationOrder": self.original_data["ImplementationOrder"]
            }
            
            similarity = graph_similarity(
                    generated_implementation,
                    original_implementation,
                    alpha=0.6
            )
            self.scores["feasibility_objective"] = similarity * 10
            self.details["feasibility_objective"] = {
                    "score": similarity,
            }
        except Exception as e:
            logging.error(f"Error evaluating feasibility objective: {e}")
            self.scores["feasibility_objective"] = 0.0
            self.details["feasibility_objective"] = {"error": str(e)}
    
    def evaluate_penalties(self) -> None:
        if self.idea:
            char_count = len(self.idea)
            penalty = 0.0
            if char_count > 700:
                excess_chars = char_count - 700
                penalty += excess_chars / 100.0
            elif char_count < 300:
                deficit_chars = 300 - char_count
                penalty += deficit_chars / 100.0
            
            self.raw_scores["length_penalty"] = min(penalty, 10.0)
        else:
            self.raw_scores["length_penalty"] = 0.0
        
        if isinstance(self.idea, str):
            self.raw_scores["repetition_penalty"] = calculate_semantic_repetition(self.idea)
        else:
            self.raw_scores["repetition_penalty"] = 0.0
        
        self.details["penalties"] = {
            "text_length": len(self.idea),
            "length_penalty": self.raw_scores["length_penalty"],
            "repetition_penalty": self.raw_scores["repetition_penalty"]
        }
     
    def calculate_semantic_repetition(text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?。！？]', text) if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.0
        
        try:
            if embedding_model is None:
                logging.warning("Embedding model not available, cannot calculate semantic repetition.")
                return 0.0
            
            sentence_embeddings = embedding_model.encode(sentences)
            similarity_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
            
            upper_triangle = []
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    upper_triangle.append(similarity_matrix[i][j])
            
            if not upper_triangle:
                return 0.0
            
            avg_similarity = np.mean(upper_triangle)
            penalty = max(0, (avg_similarity - 0.2) * 10)
            
            return min(penalty, 10.0)
        
        except Exception as e:
            return 0.0
    
    def LLM_multi_rounds(self, llm_judges):
        try:            
            context = get_context_from_data(self.original_data)
            
            evaluation_results = compare_ideas_with_voting(
                original_idea_data=self.original_data,
                generated_idea_data=self.generated_data,
                context=context,
                judge_models=llm_judges
            )
            
            summary = {
                "evaluation_details": evaluation_results,
                "timestamp": datetime.now().isoformat()
            }
            
            self.scores["novelty_subjective"] =evaluation_results["final_results"]["novelty"]["res"]
            self.scores["effectiveness_subjective"] =evaluation_results["final_results"]["effectiveness"]["res"]
            self.scores["detailedness_subjective"] =evaluation_results["final_results"]["detailedness"]["res"]
            self.scores["feasibility_subjective"] =evaluation_results["final_results"]["feasibility"]["res"]
            return {
                "success": True,
                "result": summary
            }
        except Exception as e:
            logging.error(f"Error in LLM_multi_rounds: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def merge_scores(self) -> None:
        self.scores["novelty_objective"] = (
            0.5 * self.raw_scores["novelty_similarity"] +
            0.5 * self.raw_scores["cutting_edge"]
        )
        
        self.scores["detailedness_objective"] = (
            0.2 * self.raw_scores["completeness"] +
            0.4 * (10 - self.raw_scores["repetition_penalty"]) +
            0.4 * (10 - self.raw_scores["length_penalty"])
        )
        
    
    def calculate_final_score(self, llm_judges: List[str]) -> Dict[str, Any]:
        self.LLM_multi_rounds(llm_judges)
        self.evaluate_novelty_objective()
        self.evaluate_effectiveness_objective()
        self.evaluate_completeness()
        self.evaluate_feasibility_objective()
        self.evaluate_penalties()
        self.merge_scores()

        self.idea_dict.update({
            "effectiveness_objective": float(self.scores["effectiveness_objective"])*10,
            "novelty_objective": float(self.scores["novelty_objective"])*10,
            "detailedness_objective": float(self.scores["detailedness_objective"])*10,
            "feasibility_objective": float(self.scores["feasibility_objective"])*10,
            "effectiveness_subjective": 100 if self.scores["effectiveness_subjective"] == 'win' else 0,  
            "novelty_subjective": 100 if self.scores["novelty_subjective"] == 'win' else 0,
            "detailedness_subjective": 100 if self.scores["detailedness_subjective"] == 'win' else 0,
            "feasibility_subjective": 100 if self.scores["feasibility_subjective"] == 'win' else 0,
        })
        self.idea_dict.update({
            "effectiveness": (self.idea_dict["effectiveness_objective"]+self.idea_dict["effectiveness_subjective"])/2,
            "novelty": (self.idea_dict["novelty_objective"]+self.idea_dict["novelty_subjective"])/2,
            "detailedness": (self.idea_dict["detailedness_objective"]+self.idea_dict["detailedness_subjective"])/2,
            "feasibility": (self.idea_dict["feasibility_objective"]+self.idea_dict["feasibility_subjective"])/2,
            
        })
        self.idea_dict["final_score"] = (self.idea_dict["effectiveness"]+self.idea_dict["novelty"]+self.idea_dict["detailedness"]+self.idea_dict["feasibility"])/4

        return self.idea_dict


def evaluate_single_idea(ques_dict):
    try:
        evaluator = ImprovedIdeaEvaluator(ques_dict)
        evaluation_result = evaluator.calculate_final_score(llm_judges=JUDGE_MODELS)
        output=evaluation_result
        return output
        
    except Exception as e:
        logging.error(f"evaluation error: {e}")
        output = {
            "error": str(e),
            "final_score": 0.0
        }
        return output

def main():
    input_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}.json")
    with open(input_path, 'r', encoding='utf-8') as f:
        model_answers = json.load(f)
    
    logging.info(f"find {len(model_answers)} ideas to evaluate.")
    
    inp_list = [{'ques_dict': ques} for ques in model_answers]
    out_list = multi_thread(inp_list, evaluate_single_idea, 100)
    
    output_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_list, f, indent=4, ensure_ascii=False)
    
    successful_evaluations = [item for item in out_list if "error" not in item]
    
    if successful_evaluations:
        avg_effectiveness_objective = np.mean([item["effectiveness_objective"] for item in successful_evaluations])
        avg_novelty_objective = np.mean([item["novelty_objective"] for item in successful_evaluations])
        avg_detailedness_objective = np.mean([item["detailedness_objective"] for item in successful_evaluations])
        avg_feasibility_objective = np.mean([item["feasibility_objective"] for item in successful_evaluations])

        avg_effectiveness_subjective = np.mean([item["effectiveness_subjective"] for item in successful_evaluations])
        avg_novelty_subjective = np.mean([item["novelty_subjective"] for item in successful_evaluations])
        avg_detailedness_subjective = np.mean([item["detailedness_subjective"] for item in successful_evaluations])
        avg_feasibility_subjective = np.mean([item["feasibility_subjective"] for item in successful_evaluations])
        
        effectiveness_score = np.mean([item["effectiveness"] for item in successful_evaluations])
        novelty_score = np.mean([item["novelty"] for item in successful_evaluations])
        detailedness_score = np.mean([item["detailedness"] for item in successful_evaluations])
        feasibility_score = np.mean([item["feasibility"] for item in successful_evaluations])

        avg_final_score =  np.mean([item["final_score"] for item in successful_evaluations])

        
        meta_evaluation = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(out_list),
            "successful_evaluations": len(successful_evaluations),
            "average_scores": {
                "final_score": avg_final_score,
                "effectiveness": effectiveness_score,
                "novelty": novelty_score,
                "detailedness": detailedness_score,
                "feasibility": feasibility_score,
                "details":{
                    "effectiveness_objective": avg_effectiveness_objective,
                    "effectiveness_win_rate": avg_effectiveness_subjective,
                    "novelty_objective": avg_novelty_objective,
                    "novelty_win_rate": avg_novelty_subjective,
                    "detailedness_objective": avg_detailedness_objective,
                    "detailedness_win_rate": avg_detailedness_subjective,
                    "feasibility_objective": avg_feasibility_objective,
                    "feasibility_win_rate": avg_feasibility_subjective,
                }
            },
            "judge_models": JUDGE_MODELS,
        }
    else:
        meta_evaluation = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(out_list),
            "successful_evaluations": 0,
            "failed_evaluations": len(out_list),
            "error": "All evaluations failed"
        }
    
    meta_output_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}{discipline}_meta_evaluation.json")
    with open(meta_output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_evaluation, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Total Evaluations: {len(out_list)}")
    print(f"Successful: {len(successful_evaluations)}")
    
    if successful_evaluations:
        print(f"\nAverage Scores:")
        print(f"  Final Score: {avg_final_score:.2f}")

        print(f"  Effectiveness: {effectiveness_score:.2f}")
        print(f"        Effectiveness Objective: {avg_effectiveness_objective:.2f}")
        print(f"        Effectiveness Subjective Win Rate: {avg_effectiveness_subjective:.2f}%")
        
        print(f"  Novelty: {novelty_score:.2f}")
        print(f"        Novelty Objective: {avg_novelty_objective:.2f}")
        print(f"        Novelty Subjective Win Rate: {avg_novelty_subjective:.2f}%")
        
        print(f"  Detailedness: {detailedness_score:.2f}")
        print(f"        Detailedness Objective: {avg_detailedness_objective:.2f}")
        print(f"        Detailedness Subjective Win Rate: {avg_detailedness_subjective:.2f}%")

        
        print(f"  Feasibility: {feasibility_score:.2f}")
        print(f"        Feasibility Objective: {avg_feasibility_objective:.2f}")
        print(f"        Feasibility Subjective Win Rate: {feasibility_score:.2f}%")
    
    print(f"\nResults saved to:")
    print(f"  {output_path}")
    print(f"  {meta_output_path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()