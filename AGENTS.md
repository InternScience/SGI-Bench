# SGI-Bench-Agents 项目上下文

本文档记录当前仓库的项目结构、运行方式、评测流程、HuggingFace 数据集结构与后续 agent 协作注意事项。后续在本仓库工作时，优先遵循这里的上下文。

## 基本定位

- 项目名称：SGI-Bench，论文/项目主题为“Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows”。
- 目标：用科学家对齐的工作流评测大模型的 Scientific General Intelligence，覆盖科学探究循环中的 Deliberation、Conception、Action、Perception。
- 任务族：
  - Task 1 Scientific Deep Research：多跳检索、综合与推理。
  - Task 2 Idea Generation：开放式科研想法生成与多维评估。
  - Task 3.1 Dry Experiment：代码补全、运行单元测试、评分。
  - Task 3.2 Wet Experiment：实验流程/动作序列生成与结构化匹配。
  - Task 4 Experimental Reasoning：带图片的多选题与推理有效性评估。
- 学科覆盖：`astronomy`、`chemistry`、`earth`、`energy`、`information`、`life`、`material`、`mathematics`、`neuroscience`、`physics`。

## 当前工作环境

- 用户要求使用 `conda activate agent` 的环境。已在该环境验证：
  - Python 路径：`/mnt/shared-storage-user/xuwanghan/conda_env/agent/bin/python`
  - `datasets==4.4.1`
  - `huggingface_hub==0.35.1`
- 在当前沙箱里，HuggingFace 默认缓存 `/root/.cache` 不可写。下载/采样数据前应设置：

```bash
conda activate agent
export HF_HOME=/tmp/sgi_hf_home_agent
export HF_HUB_CACHE=/tmp/sgi_hf_home_agent/hub
export HF_DATASETS_CACHE=/tmp/sgi_hf_cache_agent
```

- 运行评测脚本通常还需要：

```bash
export OPENAI_API_KEY="xxxxx"
export OPENAI_BASE_URL="xxxxx"
```

- 环境变量维护原则：`.env` 和 `.env.example` 只保留项目实际读取的最小必要键。当前只使用 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`，不要再添加 `key`、`url` 等别名字段，避免配置口径分散。
- `OPENAI_BASE_URL` 应填写 OpenAI-compatible API base URL；对于 New API 网关通常需要包含 `/v1`，不要填网页登录根路径。
- `.env` 用于本地真实评测，不提交；`.env.example` 可提交，但只能放占位值，不写真实密钥。当前 `.gitignore` 已忽略 `.env`。
- `evaluation/utils.py` 会在导入时自动尝试读取当前目录、`evaluation/` 和仓库根目录附近的 `.env`。如果外部 shell 已经设置了同名环境变量，`.env` 不应覆盖它。
- 注意：shell 激活环境时会打印 `bash: /var/log/commands.log: Read-only file system`，目前看是命令日志写入失败，不影响 `conda activate agent` 和 Python 命令执行。

## 仓库结构

- `README.md`：项目说明、任务定义、Leaderboard、Quick Start、引用信息。
- `assets/`：README 展示图，包括 teaser、pipeline、evaluation framework、subjects、reward curves、wechat。
- `evaluation/`：主要评测脚本。
  - `requirements.txt`：通用评测环境，Python 3.13.7，包含 `datasets==4.4.1`、`openai==2.3.0`、`json_repair` 等。
  - `utils.py`：通用 LLM/VLM 封装、多线程/多进程工具、答案抽取、代码函数替换、idea 评估辅助函数、结果汇总。
  - `sgi_score.py`：读取五类任务日志并计算最终 SGI-Score。
  - `task_1_deep_research/`：Deep Research 推理与评分。
  - `task_2_idea_generation/`：Idea Generation 推理与评分；有独立 `idea_generation_requirements.txt`，包含 `sentence-transformers`、`networkx`、`torch` 等。
  - `task_3_dry_experiment/`：Dry Experiment 构建代码目录、生成答案、运行代码、评分；有独立 `dry_experiment_requirements.txt`。
  - `task_3_wet_experiment/`：Wet Experiment 答案生成与动作序列评分。
  - `task_4_experimental_reasoning/`：多模态 Experimental Reasoning 答案生成与评分。
- `evaluation/task_3_dry_experiment/data/`：dry experiment 的部分本地外部数据，约 32 MB，包括 Adult、MNIST 原始压缩文件、3D user study zip。
- `local_researchharness_integration/`：ResearchHarness/SciEval 适配层。
  - `agent_runner.py`：根据 JSON config 构建 agent 和 dataset，运行 infer/eval，保存轨迹、预测与 summary。
  - `scieval/agents/research_harness_agent.py`：通过外部 `run_agent_path` 启动 agent，读取 `outputs/answer.txt`、`_session_state.json` 或 trace 中的最终答案。
  - `scieval/dataset/SGI_Bench_1_0/deep_research.py`：SciEval 版 DeepResearch 数据集与评分逻辑，使用 `InternScience/SGI-DeepResearch-Gold`。

## HuggingFace 数据集

代码中直接使用以下数据集，均有 `test` split。已在 `agent` 环境下用 streaming 方式采样 1 条样本确认字段结构；多模态数据只保留图片列 schema，采样时移除了 `images` 和 `step_images` 以避免完整解码图片。

### `InternScience/SGI-DeepResearch`

- 用途：Task 1。
- 加载位置：`evaluation/task_1_deep_research/step_1_get_answer.py`。
- 字段：
  - `idx`: 字符串，如 `SGI_DeepResearch_0000`
  - `question`: 问题文本
  - `steps`: 参考推理步骤列表
  - `answer`: 标准答案字符串
  - `discipline`: 学科
  - `direction`: 细分研究方向
  - `type`: 题型/能力类型
- 样本观察：首条是 astronomy/gravitational wave 相关数值题，`steps` 长度为 6，`answer` 为字符串形式数值。

### `InternScience/SGI-IdeaGeneration`

- 用途：Task 2。
- 加载位置：`task_2_idea_generation/step_1_get_answer.py` 与 `step_2_score.py`。
- 字段：
  - `idx`, `question`, `discipline`, `direction`
  - 上下文字段：`related_work`、`challenge`、`limitation`、`motivation`、`task_objective`、`existing_solutions`
  - 参考 idea 字段：`keywords`、`core_idea`、`implementation_steps`、`implementation_order`、`data`、`evaluation_metrics`、`expected_outcome`、`related_work_test`
- 样本观察：首条为 life/protein structure prediction，`question` 长约 5.6k 字符；`related_work`、`implementation_steps`、`evaluation_metrics` 等是字符串化 dict，脚本中用 `ast.literal_eval` 解析。

### `InternScience/SGI-DryExperiment`

- 用途：Task 3.1。
- 加载位置：`task_3_dry_experiment/step_1_build.py` 与 `step_2_get_answer.py`。
- 字段：
  - `idx`, `question`, `discipline`, `direction`
  - `data_code`: 数据准备代码
  - `main_code`: 完整参考主代码
  - `incomplete_main_code`: 待补全主代码
  - `incomplete_functions`: 需要补全的函数名列表
  - `unit_test_0_data` 到 `unit_test_4_data`: 五组测试数据生成代码
  - `unit_test_0_output` 到 `unit_test_4_output`: 五组参考输出
  - `function_type`: 函数类别
  - `runtime`: 参考运行时间，HF schema 为 `float16`
- 样本观察：首条为 gravitational wave 相关代码题，待补全函数包括 `calculate_chirp_mass`、`estimate_final_mass_spin`；`question` 约 16.8k 字符。

### `InternScience/SGI-WetExperiment`

- 用途：Task 3.2。
- 加载位置：`task_3_wet_experiment/step_1_get_answer.py`。
- 字段：
  - `idx`, `question`, `action_pool`, `answer`, `discipline`, `direction`
- 样本观察：首条为 life/tumor immunotherapy，`action_pool` 是可用动作集合文本，`answer` 是形如 `var = <Action>(...)` 的结构化实验流程。

### `InternScience/SGI-Reasoning`

- 用途：Task 4。
- 加载位置：`task_4_experimental_reasoning/step_1_get_answer.py` 与 `step_2_score.py`。
- 字段：
  - `idx`, `question`
  - `images`: `List(Image(mode=None, decode=True))`
  - `options`: 选项文本列表
  - `steps`: 参考推理步骤列表，可能包含 `<img>` 占位
  - `step_images`: `List(Image(mode=None, decode=True))`
  - `answer`: `int32`，脚本中转为 `A/B/C...`
  - `image_type`, `discipline`, `direction`, `type`
- 样本观察：首条为 life/medical imaging algorithm 方向，`options` 长度为 10，`answer` 为 `1`，即选项 B。
- 注意：在 `agent` 环境中采样该数据集时，打印完样本摘要后 Python 退出阶段出现过一次 `PyGILState_Release` fatal error，数据结构已成功输出，但后续若大量加载图片应单独验证环境稳定性。

## 评测流程

所有 README 中的命令默认从 `evaluation/` 目录执行。

### Task 1 Deep Research

- `step_1_get_answer.py`：
  - 加载 `SGI-DeepResearch`。
  - 对每条样本把 `question` 加上 `<answer>...</answer>` 输出要求后调用 `LLM(model_name)`。
  - 用 `extract_final_answer` 抽取最终答案；必要时用 `AnswerPaser` 规范化。
  - 输出到 `task_1_deep_research/logs/{model}{discipline}.json`。
- `step_2_score.py`：
  - exact match：比较标准答案、原始模型答案、parser 后答案。
  - answer-level LLM judge：新增 `llm_judge` 二值指标，参考 `local_researchharness_integration/scieval/dataset/SGI_Bench_1_0/deep_research.py` 的语义判定 prompt。它用于弥补 exact match 过严的问题，例如参考答案为 `62.0`、模型输出为 `62%`，在题目上下文说明二者等价时应由 judge 判为正确。
  - `llm_judge_reason` 保存 answer-level judge 的简短理由。`exact_match == 1` 时直接令 `llm_judge = 1`，避免额外调用 judge。
  - step-level accuracy：用 `o4-mini` 作为 judge，对模型推理步骤与参考步骤逐步比较。
  - 原逐步推理 judge 的原始结果保存为 `step_llm_judge`，避免与 answer-level `llm_judge` 含义混淆。
  - 小规模真实测试经验：可构造 1 条临时日志 `evaluation/task_1_deep_research/logs/__task1_real_smoke__['all'].json`，从 `evaluation/` 目录运行 `python task_1_deep_research/step_2_score.py __task1_real_smoke__`。测试后必须删除临时日志和 `__pycache__`。已验证过一条真实 LLM 调用：`exact_match = 0`、`llm_judge = 1`、`step_level_acc = 0.666...`。

### Task 2 Idea Generation

- `step_1_get_answer.py`：
  - 加载 `SGI-IdeaGeneration`。
  - 要求模型生成 JSON 风格 proposal，目标字段包括 `Idea`、`ImplementationSteps`、`ImplementationOrder`、`Dataset`、`EvaluationMetrics`、`ExpectedOutcome`。
  - `parse_generated_idea` 会尝试从 markdown JSON block、纯 JSON 或正则文本中解析结构。
- `step_2_score.py`：
  - 使用 `SentenceTransformer('all-MiniLM-L6-v2')` 计算客观相似度/重复度。
  - 使用多个 judge 模型做正反位置投票，默认 `gpt-5.1-2025-11-13`、`gemini-3-pro-preview`、`anthropic/claude-sonnet-4.5`。
  - 汇总 `effectiveness`、`novelty`、`detailedness`、`feasibility` 与 `final_score`。

### Task 3.1 Dry Experiment

- `step_1_build.py`：
  - 加载 `SGI-DryExperiment`。
  - 为每条样本和 5 个 unit test 创建 `task_3_dry_experiment/codes/{idx}/unit_test_{n}/`。
  - 写入 `data_en.py` 与 `main_en.py`，复制部分本地数据目录，并在 `dryexp` conda 环境中运行数据初始化脚本。
- `step_2_get_answer.py`：
  - 让模型补全 `incomplete_functions`。
  - 用 `replace_function` 将模型函数替换回 `incomplete_main_code`。
  - 每个 unit test 写出 `main_[{model}].py`。
- `step_3_run_code.py`：
  - 在 `dryexp` conda 环境运行每个 unit test 的模型代码，记录 stdout/stderr、return code 和 runtime。
- `step_4_score.py`：
  - 先做精确输出匹配。
  - 若无运行错误但输出不完全一致，用 `o4-mini` judge 判断是否可接受。
  - 计算 `PassAll@5`、`PassAll@3`、`PassAll@1`、`AET`、`SER`。

### Task 3.2 Wet Experiment

- `step_1_get_answer.py`：
  - 加载 `SGI-WetExperiment`。
  - 要求模型在 `<answer>...</answer>` 内输出形如 `var = <Action>(args...)` 的流程。
- `step_2_score.py`：
  - `parse_experiment_steps` 解析动作、输入参数、输出变量。
  - `compare_exp_steps` 用 Kendall tau 风格的顺序相似度与参数传递准确率评分。
  - 最终分数为 action sequence similarity 与 parameter accuracy 的平均。

### Task 4 Experimental Reasoning

- `step_1_get_answer.py`：
  - 加载 `SGI-Reasoning`。
  - 构造多选题 prompt，传入 `images`，要求最终答案为 `\boxed{A}` 格式。
  - 保存 JSON 前删除不可序列化的 `images`、`step_images`。
- `step_2_score.py`：
  - 重新从 HF 数据集中恢复图片列。
  - `MCA`：从 `\boxed{...}` 抽取选项并与标准答案比较。
  - `RV`：用 VLM judge 根据参考推理步骤给 0-10 分，再除以 10。

### SGI-Score

- `evaluation/sgi_score.py` 读取五个任务日志：
  - Deep Research：`exact_match`
  - Idea Generation：`final_score`
  - Dry Experiment：`PassAll@5`
  - Wet Experiment：`final_score`
  - Experimental Reasoning：`MCA`
- 最终 `SGI_Score` 是五项平均。

## 重要实现细节

- `utils.LLM` 和 `utils.VLM` 会在初始化时尝试 `chat.completions`，失败后尝试 `responses` API。初始化会真实发起一次测试请求。
- OpenAI-compatible endpoint 不一定严格返回 OpenAI SDK 对象；有些网关会直接返回字符串，`evaluation/utils.py` 中的响应解析需要兼容 SDK 对象、dict 和 string。若 `OPENAI_BASE_URL` 填成 New API 的网页登录根路径，调用会返回 HTML 页面而非模型输出；应改为 `/v1` API base。
- `AnswerPaser` 拼写如此，内部固定使用 `gpt-4.1-mini` 做结构化解析。
- 多任务脚本普遍用 `multi_thread(..., max_workers=100)` 或 `multi_process(..., max_workers=100)`，实际运行前要评估 API rate limit、CPU 与内存。
- discipline 过滤参数形如字符串 `"['all']"` 或 `"['physics']"`，代码中用 `eval` 得到列表。外部传参时需谨慎。
- 脚本默认相对路径从 `evaluation/` 目录解析；若在仓库根目录直接运行，部分路径可能不匹配。
- dry experiment 的运行依赖 `conda run -n dryexp python ...`，需要提前创建 `dryexp` 环境。
- idea generation 的评分会下载/加载 sentence-transformer 模型；如果缓存目录不可写，同样需要设置 HF cache 环境变量。

## 输出与日志约定

- 各任务生成结果默认写入：
  - `evaluation/task_1_deep_research/logs/`
  - `evaluation/task_2_idea_generation/logs/`
  - `evaluation/task_3_dry_experiment/logs/`
  - `evaluation/task_3_wet_experiment/logs/`
  - `evaluation/task_4_experimental_reasoning/logs/`
- dry experiment 还会生成大量代码目录：
  - `evaluation/task_3_dry_experiment/codes/{idx}/unit_test_{n}/`
- 这些目录可能很大，提交前应确认 `.gitignore` 是否覆盖 logs/codes/cache。当前工作区已有用户侧 `.gitignore` 未提交改动，本次上下文整理不应覆盖它。

## 后续协作准则

- 在本仓库执行 Python/HF 相关命令时，先进入：

```bash
conda activate agent
```

- 如果需要真正运行 README Quick Start 中的完整评测，应切到 `evaluation/` 目录，并确认 API key、base url、judge model、rate limit、HF cache、dryexp/idea 环境都已准备好。
- 不要把 HuggingFace 全量数据、图片解码缓存、评测 logs、dry experiment 生成代码目录误提交。
- 修改评测逻辑前，先确认对应任务的 HF 字段名与脚本中的硬编码字段一致，尤其是 idea 的字符串化 dict、reasoning 的图片列、dry experiment 的五组 unit test 字段。
- 改动要保持任务边界清晰。给 Task 1 增加新指标时，不要顺手修改 `evaluation/sgi_score.py` 的最终汇总口径，除非用户明确要求。当前 SGI-Score 仍按原逻辑使用 Deep Research 的 `exact_match`。
- 真实调用测试必须小规模，优先 1 条临时日志；不要直接跑完整 HF test split。测试产物包括临时 logs、`__pycache__`、临时 pyc 都要清理。
- 维护 `.env.example` 时只放最小必要占位字段：`OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。不要添加同义别名、个人配置、真实密钥或无关服务参数。
