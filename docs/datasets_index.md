# 数据集集成索引

本文档汇总所有已集成和候选数据集的权威信息，便于后续接入和维护。

---

## 已集成数据集（13 个）

### 第一批（数学/知识基线）

| 数据集 | 适配器 | 配置 | 规模 | 评估器 | 状态 |
|--------|--------|------|------|--------|------|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | `adapter/gsm8k/` | `configs/gsm8k.yaml` | 8.5k | accuracy | ✅ 完成 |
| [MATH](https://huggingface.co/datasets/HuggingFaceH4/MATH) | `adapter/math/` | `configs/math.yaml` | 12.5k | amo_boxed | ✅ 完成 |
| [MMLU](https://huggingface.co/datasets/EleutherAI/mmlu) | `adapter/mmlu/` | `configs/mmlu.yaml` | 14.2k | accuracy | ✅ 完成 |

### 第二批（进阶基准）

| 数据集 | 适配器 | 配置 | 规模 | 评估器 | 状态 |
|--------|--------|------|------|--------|------|
| [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | `adapter/mmlu_pro/` | `configs/mmlu_pro.yaml` | 12k | accuracy | ✅ 完成 |
| [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | `adapter/gpqa/` | `configs/gpqa.yaml` | 448 | accuracy | ✅ 完成 |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | `adapter/ifeval/` | `configs/ifeval.yaml` | 500+ | ifeval_rule | ✅ 完成 |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | `adapter/humaneval/` | `configs/humaneval.yaml` | 164 | humaneval_pass | ✅ 完成 |

### 原有适配器

| 数据集 | 适配器 | 配置 | 规模 | 评估器 |
|--------|--------|------|------|--------|
| ChineseSimpleQA | `adapter/chinese_simpleqa/` | `configs/test.yaml` | ~5k | accuracy |
| C-Eval | `adapter/ceval/` | - | 13.9k | accuracy |
| OlympiadBench | `adapter/olympiadbench/` | - | 8.5k | amo_boxed |
| AMO-Bench | `adapter/amo_bench/` | - | 50 | amo_boxed |
| AlignBench | `adapter/alignbench/` | - | 100 | alignbench_judge |
| WritingBench | `adapter/writingbench/` | - | ~200 | writingbench_score |
| AIME 2025 | `adapter/aime2025/` | - | 30 | accuracy |

---

## 候选数据集（18 个，按优先级）

### ⭐⭐⭐ 首选（已完成）
- ✅ GSM8K
- ✅ MATH
- ✅ MMLU

### ⭐⭐ 第二批（已完成）
- ✅ MMLU-Pro
- ✅ GPQA
- ✅ IFEval
- ✅ HumanEval

### ⭐ 第三批（待接入）

| # | 数据集 | 领域 | 规模 | 难点 | 接入建议 |
|---|--------|------|------|------|----------|
| 8 | [MMMU](https://huggingface.co/datasets/MMMU/MMMU) | 多模态 | 11.5k | 需图像输入 | 需扩展框架支持多模态 |
| 9 | [MATH-Vista](https://huggingface.co/datasets/LMMs-Lab/MathVista) | 多模态数学 | 3.9k | 需图像输入 | 需扩展框架支持多模态 |
| 10 | [HLE](https://huggingface.co/datasets/cais/hle) | 全学科前沿 | 2.5k | 需 HF 登录 | 需处理认证 |
| 11 | [SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | 软件工程 | 2.3k | 需代码沙箱 | 需容器化执行环境 |
| 12 | [BFCL](https://huggingface.co/datasets/ShishirSilBFCL/bfcl_v3) | 工具调用 | 2.7k | 需 AST 解析 | 可扩展评估器类型 |

### ⭐ 观察级（轻量基线）

| # | 数据集 | 领域 | 规模 | 接入难度 | 备注 |
|---|--------|------|------|----------|------|
| 13 | [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | 常识 | 10k | 低 | 仿 MMLU 模式 |
| 14 | [AGIEval](https://huggingface.co/datasets/NLP2CTEvals/AGIEval) | AGI 能力 | 1.7k | 低 | 中英文混合 |
| 15 | [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | 幻觉 | 817 | 中 | 需 LLM 判分 |
| 16 | [CLUEWSC](https://huggingface.co/datasets/clue/clue) | 中文推理 | 5k | 低 | 二选一选择题 |
| 17 | [WebArena](https://github.com/web-arena-x/webarena) | 网页代理 | 812 | 极高 | 需浏览器环境 |
| 18 | [GSM-Plus](https://huggingface.co/datasets/datalab-to/GSM-Plus) | 抗干扰数学 | 9k+ | 低 | 基于 GSM8K |

---

## 接入流程

### 标准流程（文本类数据集）

```bash
# 1. 创建适配器目录
mkdir adapter/<slug>/

# 2. 编写适配器（继承 BaseDataset）
# 参考: adapter/mmlu/__init__.py

# 3. 编写评估器（如需要）
# 参考: adapter/ifeval/__init__.py (ifeval_rule)

# 4. 创建配置文件
cat > configs/<slug>.yaml << 'EOF'
tasks:
  - name: <slug>
    type: standard
    dataset:
      name: <Name>
      params:
        data_path: "data/<slug>/test.jsonl"
    prompt_builder:
      name: qa_builder
    model:
      name: MiniMax
      params:
        generation_config:
          temperature: 0.0
    evaluators:
      - name: accuracy
    output_path: results/<slug>
EOF

# 5. 运行评测
python cli/main.py --config configs/<slug>.yaml
```

### 特殊流程

**需 HF 登录**（GPQA、HLE）：
```bash
huggingface-cli login
# 然后手动下载数据到 data/<slug>/
```

**多模态**（MMMU、MATH-Vista）：
- 需扩展框架支持图像输入
- 需多模态模型（如 GPT-4V、Claude 3）

**代码执行**（HumanEval、SWE-bench）：
- 需 Docker 沙箱环境
- 需 pytest 依赖

**浏览器代理**（WebArena）：
- 需浏览器自动化（Playwright/Selenium）
- 需服务部署

---

## 常见问题

### Q: 如何添加新评估器？
参考 `adapter/ifeval/__init__.py` 中的 `IFEvalRuleEvaluator`，实现 `evaluate(pred, item) -> dict` 方法。

### Q: 如何处理多选项问题？
参考 `adapter/mmlu/__init__.py`，将 choices list 映射为 A/B/C/D 字母。

### Q: 如何支持 boxed 答案？
参考 `adapter/math/__init__.py`，使用正则提取 `\boxed{}` 内容。

---

## 更新日志

- 2026-10-03: 完成第一批（GSM8K, MATH, MMLU）和第二批（MMLU-Pro, GPQA, IFEval, HumanEval）接入
- 2026-10-02: 修复 7 个框架 bug，新增测试覆盖
