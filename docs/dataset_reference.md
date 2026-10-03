# 数据集权威信息汇总

本文档汇总所有已集成数据集的下载地址、论文、榜单链接。

---

## 已集成数据集详情

### 1. GSM8K

| 项目 | 信息 |
|------|------|
| 全称 | Grade School Math 8K |
| 来源 | https://huggingface.co/datasets/openai/gsm8k |
| 论文 | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (NeurIPS 2021) |
| 仓库 | https://github.com/openai/grade-school-math |
| 规模 | 8,500 题（7,473 train + 1,000 test） |
| 领域 | 小学数学应用题 |
| 难度 | 低 |
| Schema | `question`, `answer`（含 #### 标记） |
| 评估 | 精确匹配最终数字 |
| 已集成 | ✅ |

### 2. MATH

| 项目 | 信息 |
|------|------|
| 全称 | MATH: Measuring Mathematical Problem Solving |
| 来源 | https://huggingface.co/datasets/HuggingFaceH4/MATH |
| 论文 | [MATH](https://arxiv.org/abs/2103.03094) (NeurIPS 2021) |
| 仓库 | https://github.com/hendrycks/math |
| 规模 | 12,500 题（5,000 train + 7,500 test） |
| 领域 | 数学竞赛（代数/几何/数论等 8 类） |
| 难度 | 中-高（Level 1-5） |
| Schema | `problem`, `solution`, `answer`（\boxed{}）, `topic`, `level` |
| 评估 | boxed 提取 + 数值匹配 |
| 已集成 | ✅ |

### 3. MMLU

| 项目 | 信息 |
|------|------|
| 全称 | Measuring Massive Multitask Language Understanding |
| 来源 | https://huggingface.co/datasets/EleutherAI/mmlu |
| 论文 | [MMLU](https://arxiv.org/abs/2009.03300) |
| 仓库 | https://github.com/EleutherAI/lm-evaluation-harness |
| 规模 | 14,269 题，57 学科 |
| 领域 | 多学科知识问答 |
| 难度 | 中 |
| Schema | `question`, `choices`（list）, `answer`（index）, `subject` |
| 评估 | 精确匹配选项字母 |
| 已集成 | ✅ |

### 4. MMLU-Pro

| 项目 | 信息 |
|------|------|
| 全称 | MMLU-Pro: A More Robust and Challenging Multi-Task Benchmark |
| 来源 | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| 论文 | [MMLU-Pro](https://arxiv.org/abs/2406.01574) (NeurIPS 2024 D&B) |
| 仓库 | https://github.com/TIGER-AI-Lab/MMLU-Pro |
| 榜单 | https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro |
| 规模 | 12,000 题，14 领域 |
| 领域 | 多学科知识（10 选项） |
| 难度 | 高 |
| Schema | `question`, `options`（list of 10）, `answer`（index）, `subject` |
| 评估 | 精确匹配选项字母 |
| 已集成 | ✅ |

### 5. GPQA

| 项目 | 信息 |
|------|------|
| 全称 | GPQA: A Graduate-Level Google-Proof Q&A Benchmark |
| 来源 | https://huggingface.co/datasets/Idavidrein/gpqa |
| 论文 | [GPQA](https://arxiv.org/abs/2311.12022) |
| 仓库 | https://github.com/idavidrein/gpqa |
| 规模 | 448 题（Diamond 198 / Balanced 140 / All 448） |
| 领域 | 研究生级科学（生物/物理/化学） |
| 难度 | 极高 |
| Schema | `Question`, `A/B/C/D`, `Correct Answer`, `Explanation`, `Domain` |
| 评估 | 精确匹配选项字母 |
| 已集成 | ✅ |
| 注意 | 需 HF 登录同意条款 |

### 6. IFEval

| 项目 | 信息 |
|------|------|
| 全称 | Instruction-Following Evaluation |
| 来源 | https://huggingface.co/datasets/google/IFEval |
| 论文 | [Instruction-Following Evaluation](https://arxiv.org/abs/2311.07911) |
| 仓库 | https://github.com/google/IFEval |
| 规模 | 500+ 指令，25 类约束 |
| 领域 | 指令跟随 |
| 难度 | 中 |
| Schema | `prompt`, `key`（约束类型）, `instruction_id_list` |
| 评估 | 规则引擎自动判分 |
| 已集成 | ✅ |

### 7. HumanEval

| 项目 | 信息 |
|------|------|
| 全称 | HumanEval: Evaluating Large Language Models Trained on Code |
| 来源 | https://huggingface.co/datasets/openai/openai_humaneval |
| 论文 | [HumanEval](https://arxiv.org/abs/2107.03374) |
| 仓库 | https://github.com/openai/human-eval |
| 规模 | 164 题 |
| 领域 | Python 函数补全 |
| 难度 | 高 |
| Schema | `task_id`, `prompt`, `canonical_solution`, `test`, `setup_code`, `entry_point` |
| 评估 | Pass@k（代码执行 + 单元测试） |
| 已集成 | ✅ |
| 注意 | 需代码执行沙箱 |

### 8. OlympiadBench

| 项目 | 信息 |
|------|------|
| 全称 | OlympiadBench: A Challenge Benchmark for Olympiad-Level Reasoning |
| 来源 | https://huggingface.co/datasets/OpenBMB/OlympiadBench |
| 论文 | [OlympiadBench](https://arxiv.org/abs/2402.14008) |
| 仓库 | https://github.com/OpenBMB/OlympiadBench |
| 规模 | 8,476 题（含 IMO 子集） |
| 领域 | 数学/物理竞赛 |
| 难度 | 极高 |
| Schema | `id`, `subfield`, `question`, `solution`, `final_answer`, `answer_type` |
| 评估 | boxed 提取 + 数值匹配 |
| 已集成 | ✅ |

### 9. C-Eval

| 项目 | 信息 |
|------|------|
| 全称 | C-Eval: A Multi-Level Multi-Domain Chinese Evaluation Benchmark |
| 来源 | https://huggingface.co/datasets/ceval/ceval-exam |
| 论文 | [C-Eval](https://arxiv.org/abs/2305.08322) |
| 仓库 | https://github.com/hujie-frank/CEval |
| 规模 | 13,948 题，52 学科 |
| 领域 | 中文知识问答 |
| 难度 | 中（4 级别） |
| Schema | `question`, `A/B/C/D`, `answer`, `explanation`, `category`, `exam` |
| 评估 | 精确匹配选项字母 |
| 已集成 | ✅ |

### 10. ChineseSimpleQA

| 项目 | 信息 |
|------|------|
| 全称 | Chinese Simple Question Answering |
| 来源 | 本地数据 `E:/eval/Chinese-SimpleQA/` |
| 规模 | ~5,000 题 |
| 领域 | 中文事实性问答 |
| 难度 | 低 |
| Schema | `question`, `answer`, `primary_category`, `secondary_category` |
| 评估 | 精确匹配 |
| 已集成 | ✅ |

### 11. AlignBench

| 项目 | 信息 |
|------|------|
| 全称 | AlignBench: Evaluating Aligned LLMs in Chinese |
| 来源 | https://huggingface.co/datasets/THUDM/AlignBench |
| 论文 | [AlignBench](https://arxiv.org/abs/2312.15563) (ACL 2024) |
| 仓库 | https://github.com/THUDM/AlignBench |
| 规模 | 100 题（5 类 × 20 题） |
| 领域 | 中文对齐能力 |
| 难度 | 中 |
| Schema | `id`, `category`, `subcategory`, `question`, `reference` |
| 评估 | LLM-as-Judge（多维度） |
| 已集成 | ✅ |

### 12. WritingBench

| 项目 | 信息 |
|------|------|
| 全称 | WritingBench: Evaluating Long-form Text Generation |
| 来源 | https://huggingface.co/datasets/X-PLUG/WritingBench |
| 论文 | [WritingBench](https://arxiv.org/abs/2406.03803) (NeurIPS 2025 D&B) |
| 仓库 | https://github.com/X-PLUG/WritingBench |
| 规模 | ~200 题 |
| 领域 | 生成式写作 |
| 难度 | 中 |
| Schema | `query`, `checklist`（评分标准列表） |
| 评估 | LLM-as-Judge（checklist 逐条评分） |
| 已集成 | ✅ |

### 13. AMO-Bench

| 项目 | 信息 |
|------|------|
| 全称 | AMO-Bench: Large Language Models Struggle in High-Difficulty Math |
| 来源 | https://huggingface.co/datasets/meituan-longcat/AMO-Bench |
| 论文 | [AMO-Bench](https://arxiv.org/abs/2406.06077) |
| 仓库 | https://github.com/meituan-longcat/AMO-Bench |
| 规模 | 50 题 |
| 领域 | 高级数学推理 |
| 难度 | 极高 |
| Schema | `question`, `answer`（boxed 格式） |
| 评估 | boxed 提取 + 数值匹配 |
| 已集成 | ✅ |

### 14. AIME 2025

| 项目 | 信息 |
|------|------|
| 全称 | American Invitational Mathematics Examination 2025 |
| 来源 | 本地数据（需手动获取） |
| 论文 | 官方考试，无论文 |
| 官网 | https://www.maasociety.org/ |
| 规模 | 30 题（I/II 卷各 15 题） |
| 领域 | 数学竞赛 |
| 难度 | 高 |
| Schema | `problem`, `answer`（整数 0-999）, `year`, `number` |
| 评估 | 精确匹配整数 |
| 已集成 | ✅ |

---

## 数据集分类统计

| 类别 | 数量 | 代表数据集 |
|------|------|------------|
| 数学推理 | 5 | GSM8K, MATH, MMLU, OlympiadBench, AIME |
| 知识问答 | 3 | MMLU, C-Eval, ChineseSimpleQA |
| 代码生成 | 1 | HumanEval |
| 指令跟随 | 1 | IFEval |
| 多模态 | 0 | - |
| 工具调用 | 0 | - |
| 安全幻觉 | 0 | - |
| 中文专项 | 2 | C-Eval, AlignBench |
| 综合基准 | 2 | GPQA, WritingBench |

---

## 下载命令

```bash
# GSM8K
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir data/gsm8k

# MATH
huggingface-cli download HuggingFaceH4/MATH --repo-type dataset --local-dir data/math

# MMLU
huggingface-cli download EleutherAI/mmlu --repo-type dataset --local-dir data/mmlu

# MMLU-Pro
huggingface-cli download TIGER-Lab/MMLU-Pro --repo-type dataset --local-dir data/mmlu_pro

# GPQA（需登录）
huggingface-cli login
huggingface-cli download Idavidrein/gpqa --repo-type dataset --local-dir data/gpqa

# HumanEval
huggingface-cli download openai/openai_humaneval --repo-type dataset --local-dir data/humaneval

# IFEval
huggingface-cli download google/IFEval --repo-type dataset --local-dir data/ifeval

# OlympiadBench
huggingface-cli download OpenBMB/OlympiadBench --repo-type dataset --local-dir data/olympiadbench
```
