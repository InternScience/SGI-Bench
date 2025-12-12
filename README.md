<div align="center">

# Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows

</div>

<!-- <p align="center">
  <a href="https://internscience.github.io/SGI-Page/paper.pdf"><b>Paper</b></a> ·
  <a href="https://github.com/InternScience/SGI-Bench"><b>GitHub</b></a> ·
  <a href="https://internscience.github.io/SGI-Page/"><b>Page</b></a> ·
  <a href="https://huggingface.co/collections/InternScience/sgi-bench"><b>HuggingFace</b></a>
</p> -->

<div align="center">

[![Official Site](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://internscience.github.io/SGI-Page/)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SGI--Bench-yellow)](https://huggingface.co/collections/InternScience/sgi-bench)&#160;
[![PDF](https://img.shields.io/badge/%F0%9F%96%BC%20PDF-Paper-ff69b4)](https://internscience.github.io/SGI-Page/paper.pdf)&#160;
[![GitHub](https://img.shields.io/badge/GitHub-000000?logo=github&logoColor=white)](https://github.com/InternScience/SGI-Bench)&#160;
<!-- <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" height="21px"></a> -->

Welcome to the official repository for the SGI-Bench!

</div>

<p align="center">
  <img src="assets/teaser.png" alt="SGI Overview" width="850">
</p>

Scientist-aligned benchmark for evaluating Scientific General Intelligence (SGI) across the full inquiry cycle: Deliberation, Conception, Action, and Perception. The benchmark spans 10 disciplines and more than 1,000 expert‑curated samples inspired by Science’s 125 Big Questions, with an agentic evaluation framework and multi‑metric protocol.

---

## 🔬 What is Scientific General Intelligence (SGI)?
SGI denotes an AI system that can autonomously navigate the full, iterative cycle of scientific inquiry—Deliberation, Conception, Action, and Perception—with the versatility and proficiency of a human scientist. SGI‑Bench operationalizes this definition via four scientist‑aligned task families: deep research, idea generation, AI‑assisted experiments (dry/wet), and multimodal experimental reasoning.

---

## 🎯 Framework & Tasks

<p align="center">
  <img src="assets/pipeline.png" alt="SGI-Bench Pipeline" width="850">
</p>

- Deliberation (Deep Research): Multi‑hop retrieval, synthesis, and meta‑analysis style reasoning.
- Conception (Idea Generation): Structured ideation and multi‑dimensional comparative evaluation.
- Action (Dry/Wet Experiment): Code/simulation and lab protocol generation and verification.
- Perception (Multimodal Reasoning): Process/observation/simulation/experiment/visualization image reasoning.

Grounded in the Practical Inquiry Model (PIM), SGI‑Bench treats science as an iterative cycle linking deliberation, conception, action and perception. Under this lens, SGI captures the capacity to integrate knowledge retrieval, idea formation, action execution, and interpretation into a unified loop of inquiry.

---

## 📂 Scientist‑Aligned Data Construction

<p align="center">
  <img src="assets/subjects.png" alt="Scientist-Aligned Data Construction" width="850">
</p>

- Raw Corpus: Expert‑curated texts/images across 10 domains, inspired by Science’s 125 Big Questions.
- Question Construction: 100+ graduate/PhD annotators with continuous expert‑in‑the‑loop review.
- Data Cleaning: Rules + model checks + expert QA to ensure executability and unique answers.
- Difficulty Filtering: Removes samples solved by >50% strong LLMs to maintain high challenge.

Result: High‑fidelity, scientist‑aligned tasks that are authentic, challenging, and broadly representative.

---

## 💯 Agentic Evaluation Framework

<p align="center">
  <img src="assets/evaluation-framework.png" alt="Agentic Evaluation Framework" width="850">
</p>

- Four Stages: Question Selection → Metric Customization → Predict & Eval → Report Generation
- Tool Pool: Web search, PDF parser, Python interpreter, file reader, metric functions
- Task Metrics: EM/SLA; Implementation Similarity; PassAll@k/SER; MCA/RV
- Customizable: Add scientist‑aligned metrics (e.g., rigor, feasibility) on demand

This agent‑based stack formalizes scoring into traceable stages, improves reproducibility, mitigates evaluator–model coupling bias, and yields actionable, scientist‑aligned insights.

---

## 🚀 Test‑Time Reinforcement Learning (TTRL)

<p align="center">
  <img src="assets/grpo_reward_curves.png" alt="TTRL Training Dynamics" width="850">
</p>

- Objective: Address no‑ground‑truth idea generation by optimizing novelty at test time with online retrieval as a moving baseline.
- Reward Design:  
  R = R_format + R_novelty  
  Enforce XML format and strict structure (e.g., &lt;think&gt;, &lt;answer&gt;); reward embedding dissimilarity from retrieved works, gated by thresholds.
- Setup: GRPO on Qwen3‑8B (ms‑swift), G=8, high temperature, bfloat16, online retrieval n=4.
- Dynamics: Format reward saturates quickly; novelty steadily increases. Average novelty improved from 49.36 → 62.06 without labels.

TTRL converts open‑ended ideation into measurable test‑time optimization and extends to multi‑objective rewards (rigor, feasibility, safety, cost).

---

## 🏆 Leaderboard Highlights

| Model                 | Deep Research | Idea Generation | Dry Experiment | Wet Experiment | Experimental Reasoning | SGI-Score |
| --------------------- | ------------: | --------------: | -------------: | -------------: | ---------------------: | --------: |
| Gemini-3-Pro 🥇      | **18.48**     | 39.68           | **36.64**      | 32.45          | **41.92**              | **33.83** |
| Claude-Sonnet-4.5 🥈 | 13.84         | 43.20           | 35.79          | 30.15          | 37.80                  | 32.16     |
| Qwen3-Max 🥉         | 15.38         | 39.83           | 33.21          | 33.62          | 37.80                  | 31.97     |
| GPT-4.1               | 11.32         | 36.49           | 34.32          | **36.63**      | 38.49                  | 31.45     |
| GPT-5.2-Pro           | 15.72	        | 55.03	          | 28.04	         | 17.50	        | 39.18	                 | 31.09     |
| GPT-5                 | 14.47         | **55.40**       | 29.89          | 16.31          | 38.14                  | 30.84     |
| o3                    | 12.89         | 46.07           | 31.73          | 30.04          | 32.65                  | 30.68     |
| Claude-Opus-4.1       | 12.93         | 40.29           | 34.69          | 25.38          | 38.83                  | 30.42     |
| o4-mini               | 11.95         | 40.78           | 35.79          | 28.86          | 33.33                  | 30.14     |
| GPT-5.1               | 11.64         | 47.12           | 31.00          | 22.77          | 34.02                  | 29.31     |
| Grok-4                | 13.31         | 37.12           | 33.71          | 29.01          | 30.24                  | 28.68     |
| Qwen3-VL-235B-A22B    | 11.97         | 39.28           | 28.41          | 30.30          | 31.62                  | 28.32     |
| Gemini-2.5-Pro        | 15.09         | 39.95           | 22.51          | 22.05          | 41.24                  | 28.17     |
| Intern-S1             | 15.74         | 38.09           | 28.79          | 29.02          | 28.87                  | 28.10     |
| GPT-4o                | 7.86          | 35.95           | 26.94          | 31.31          | 32.30                  | 26.87     |
| Gemini-2.5-Flash      | 10.69         | 39.13           | 21.03          | 18.55          | 34.36                  | 24.75     |
| Llama-4-Scout         | 7.86          | 29.72           | 20.37          | 21.66          | 25.77                  | 21.08     |
| Qwen3-8B              | 8.18          | 35.78           | 18.45          | 9.96           | 23.37                  | 19.15     |
| Intern-S1-mini        | 11.06         | 36.04           | 16.97          | 12.42          | 16.84                  | 18.67     |


---

## 🔥 Quick Start

```bash
git clone https://github.com/InternScience/SGI-Bench.git
cd SGI-Bench/evaluation

export OPENAI_API_KEY="xxxxx"
export OPENAI_BASE_URL="xxxxx"

conda create -n sgi python=3.13.7
conda activate sgi
pip install -r requirements.txt
```

### 📚 Task 1 Deep Research

```bash
conda activate sgi
python task_1_deep_research/step_1_get_answer.py gpt-5.2-pro
python task_1_deep_research/step_2_score.py gpt-5.2-pro
```

### 💡 Task 2 Idea Generation

1. Install the environment dependencies for evaluating idea generation.

```bash
conda create -n idea python=3.10.18
conda activate idea
pip install -r task_2_idea_generation/idea_generation_requirements.txt
```

2. Start the evaluation.

```bash
conda activate idea
python task_2_idea_generation/step_1_get_answer.py gpt-5.2-pro
python task_2_idea_generation/step_2_score.py gpt-5.2-pro
```

### 🖥️ Task 3.1 Dry Experiment (Code Generation)

1. Install the environment dependencies for running the dry experiment code.

```bash
conda create -n dryexp python=3.10.18
conda activate dryexp
pip install -r task_3_dry_experiment/dry_experiment_requirements.txt
```

2. Create code folder and initialize data (only need to run once).

```bash
conda activate sgi
python task_3_dry_experiment/step_1_build.py
```

> Note: If some scripts time out during execution, please enter the corresponding folder and manually run the script to complete the data initialization.

3. Start the evaluation.

```bash
conda activate sgi
python task_3_dry_experiment/step_2_get_answer.py gpt-5.2-pro
python task_3_dry_experiment/step_3_run_code.py gpt-5.2-pro
python task_3_dry_experiment/step_4_score.py gpt-5.2-pro
```

### 🧪 Task 3.2 Wet Experiment (Lab Protocol)

```bash
conda activate sgi
python task_3_wet_experiment/step_1_get_answer.py gpt-5.2-pro
python task_3_wet_experiment/step_2_score.py gpt-5.2-pro
```

### 📊 Task 4 Experimental Reasoning

```bash
conda activate sgi
python task_4_experimental_reasoning/step_1_get_answer.py gpt-5.2-pro
python task_4_experimental_reasoning/step_2_score.py gpt-5.2-pro
```

### 💎 SGI-Score

```bash
conda activate sgi
python sgi_score.py gpt-5.2-pro
```

---

## 📜 Citation

```
Coming soon...
```