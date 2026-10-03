---
name: dataset-adapter
description: 读取数据集分析报告，将新数据集适配到现有评测框架中。自动生成数据集适配器、评估器、配置模板，并注册到框架。适用场景：已有数据集报告，需要将新数据集集成到评测框架、添加新数据集适配器、编写数据集代码。当用户提到"适配数据集"、"集成数据集"、"添加数据集"、"注册数据集"时使用此skill。
---

# 数据集适配器生成 Skill

本 skill 读取数据集分析报告和实际样本，将新数据集安全、可验证地适配到现有的评测框架中。生成代码只是起点：必须先核对框架基类、注册机制和评估器返回约定，再运行导入/样本级验证；不能仅凭字段名猜测完成适配。

## 输入、输出与安全边界

输入至少包括数据集报告或明确的数据文件；如果两者冲突，以实际数据为准并在结果中说明。输出通常包括 `adapter/{slug}/__init__.py`、必要时的 `evaluator.py` 和 `configs/{slug}.yaml`。生成前检查目标注册名、目录和配置是否已存在，避免静默覆盖；如存在文件，应先备份或明确提示。不要在适配器中执行原始数据里的代码或动态导入不可信模块。

## ⚠️ 已知框架问题（Issue 已创建）

| Issue | 文件 | 问题 | 状态 |
|-------|------|------|------|
| #2 | `models/test_minmax.py:10` | API Key 硬编码泄露 | 🔴 Security |
| #4 | `evaluators/base.py` | 基类返回类型与实现不一致 | 🟠 High |
| #6 | `core/leaderboard.py:23` | 非数值 metric 格式化崩溃 | 🟠 High |

> 注：之前 Issue #1（base.py死代码）和 #3（data_filter.py bug）经核实后关闭，实际代码无问题。

**建议**：生成新评估器时统一返回 `dict`（与现有所有评估器一致，Issue #4）。

## 框架适配模式

根据数据集类型，框架提供四种适配模式：

### 模式 A: 标准问答（Q&A）

适用于事实性问答、知识问答、简单推理等：
- 数据集适配器：`BaseDataset` 子类
- 提示词构建器：复用 `qa_builder` 或自定义
- 评估器：`accuracy`（精确匹配）或自定义
- 示例：`ChineseSimpleQA`, `AIME2025`, `EqBench`

### 模式 B: 多选题

适用于选择题（含 A/B/C/D 选项）：
- 数据集适配器：`BaseDataset` 子类（需将选项拼入 prompt）
- 提示词构建器：自定义多选题模板
- 评估器：`accuracy`（提取 A/B/C/D 匹配）或自定义
- 示例：`CEval`

### 模式 C: 生成式评估（LLM-as-Judge / 自定义）

适用于写作、对齐、创意等需要 LLM 打分的场景：
- 数据集适配器：`BaseDataset` 子类
- 评估器：`BaseEvaluator` 子类，实现评分逻辑
- 提示词构建器：自定义模板
- 示例：`WritingBench`, `AlignBench`

### 模式 D: 数学 boxed 答案

适用于数学竞赛类数据集，答案在 `\boxed{}` 中：
- 评估器：提取 boxed 内容后比较（可容忍格式差异）
- 示例：`AMO-Bench`, `OlympiadBench`

## 工作流程

### Step 1: 读取数据集分析报告

读取 `dataset_report/{DatasetName}_Dataset_Report.md`，提取关键信息：

| 需要提取的信息 | 查看位置 |
|--------------|---------|
| 数据字段结构 | 1.4 数据集描述 - 数据字段说明 |
| 分类体系 | 2. 能力体系 + 3. 场景体系 |
| 评测方法 | 4. 测评 - 4.1 获取模型回复 + 4.2 测评方法 |
| 提示词模板 | 4.1 获取模型回复 |

### Step 2: 查看实际数据文件

从报告中获取数据文件路径（通常为 `data/{DatasetName}/`），读取实际数据确认字段：

```python
import json
with open('data/{DatasetName}/data.jsonl', encoding='utf-8') as f:
    sample = json.loads(f.readline())
    print(json.dumps(sample, ensure_ascii=False, indent=2))
```

关键字段映射：
- 问题字段 → `DataItem.prompt`
- 答案字段 → `DataItem.reference`  
- 分类字段 → `DataItem.category`
- 难度字段 → `DataItem.difficulty`
- 其他字段 → `DataItem.metadata`

### Step 3: 生成适配器代码

创建 `adapter/{dataset_name_slug}/` 目录，包含：

```
adapter/{name}/
├── __init__.py          # 数据集适配器（注册到 Registry）
└── evaluator.py         # 评估器（可选，仅在需要自定义评估时）
```

#### 适配器模板（`__init__.py`）

```python
r"""
{dataset_name} 适配器

数据集来源: {report中的来源链接}
数据集报告: dataset_report/{dataset_name}_Dataset_Report.md
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("{registry_name}", "dataset")
class {ClassName}(BaseDataset):
    """
    {dataset_name} 数据集适配器

    数据格式:
    - {字段1}: {说明}
    - {字段2}: {说明}
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get("data_path")
        if not self.data_path:
            raise ValueError(f"{registry_name} requires 'data_path' config")
        self.dataset_name = "{registry_name}"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        # 尝试多种可能的字段名，以实际数据为准
        question = (
            data_item.get("question") or
            data_item.get("problem") or
            data_item.get("prompt", "")
        )
        answer = (
            data_item.get("answer") or
            data_item.get("reference") or
            data_item.get("solution", "")
        )

        return DataItem(
            id=self.build_id(data_item.get("id", "")),
            prompt=question,
            reference=answer,
            metadata={
                # 其他相关字段
            },
            category=[
                c for c in [
                    data_item.get("category"),
                    data_item.get("domain"),
                ] if c
            ],
            difficulty=data_item.get("difficulty", ""),
        )
```

#### 评估器模板（`evaluator.py`）

> ⚠️ 注意：当前框架评估器统一返回 `dict`（与 `AccuracyEvaluator`、`WritingBenchScoreEvaluator` 等一致，Issue #4）。

```python
"""
{dataset_name} 评估器
"""

from core.registry import Registry
from evaluators.base import BaseEvaluator


@Registry.register("{evaluator_name}", "evaluator")
class {EvalClassName}(BaseEvaluator):
    """
    {dataset_name} 评估器

    根据报告中的评测方法实现评估逻辑。
    """

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, pred: str, item) -> dict:
        """评估模型输出，返回 dict 格式 metrics"""
        reference = str(item.reference).strip()
        pred_clean = pred.strip()
        accuracy = 1.0 if pred_clean == reference else 0.0
        return {"accuracy": accuracy}
```

**数学 boxed 答案评估器示例**：

```python
import re

def _extract_boxed(self, text: str) -> str:
    """提取 \\boxed{} 中的内容"""
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"boxed\{([^}]+)\}",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None

def evaluate(self, pred: str, item) -> dict:
    ref_answer = self._extract_boxed(str(item.reference))
    pred_answer = self._extract_boxed(pred)
    if pred_answer and ref_answer:
        correct = pred_answer == ref_answer
    else:
        correct = pred.strip() == item.reference.strip()
    return {"accuracy": 1.0 if correct else 0.0}
```

### Step 4: 生成配置模板

创建 `configs/{dataset_name}.yaml` 配置示例：

```yaml
tasks:
  - name: standard
    type: standard
    dataset:
      name: {registry_name}
      params:
        data_path: "data/{dataset_name}/data.jsonl"
        limits: 10  # 测试用，可移除
    prompt_builder:
      name: qa_builder  # 或自定义
    model:
      name: MiniMax
      params:
        generation_config:
          temperature: 0.5
    evaluators:
      - name: accuracy  # 或自定义评估器
    output_path: results/{dataset_name}
```

### Step 5: 注册到框架

确保 `adapter/{name}/__init__.py` 能被加载：

1. 检查 `adapter/__init__.py` 中是否包含 `auto_import("adapter")`（已有则自动加载子目录）
2. 如果适配器是单文件模块（如 `adapter/olympiadbench.py`），需在 `adapter/__init__.py` 中添加 `from adapter import {name}`

### Step 6: 验证

运行一个测试来验证适配器工作：

```bash
python cli/main.py --config configs/{dataset_name}.yaml --log-level INFO
```

## 关键资源

| 资源 | 路径 | 说明 |
|------|------|------|
| 数据集分析报告 | `dataset_report/{name}_Dataset_Report.md` | 报告作为适配依据 |
| 已有适配器参照 | `adapter/chinese_simpleqa/` | QA 模式示例 |
| 多选题适配器 | `adapter/ceval/__init__.py` | 多选题示例 |
| 自定义评估器 | `adapter/writingbench/evaluator.py` | LLM-as-Judge 示例 |
| 数学 boxed 评估器 | `adapter/amo_bench/evaluator.py` | boxed 提取示例 |
| 框架核心类 | `core/base.py` | DataItem, ModelInput, ModelOutput |
| 评估器基类 | `evaluators/base.py` | BaseEvaluator |
| 数据集基类 | `datasets/base.py` | BaseDataset |
| 注册系统 | `core/registry.py` | Registry.register |
| 框架已知问题 | `.github/create_issues.py` | 7 个待修复 issue |

## 注意事项

- **数据和报告为准** — 数据字段以实际数据文件为准，不要假设
- **多选题特殊处理** — 在 preprocess 中把选项拼入 question
- **评估器返回 dict** — 当前框架评估器返回 `dict`，不是 `EvaluationResult`（Issue #4）
- **注册名唯一** — 注册名在框架中必须唯一，检查 `Registry.list_registered("dataset")`
- **适配器命名** — 类名建议用 `{DatasetName}Dataset`，注册名建议用 `{DatasetName}`
- **数据路径字段** — 统一使用 `data_path`（Issue #10 已关闭，建议兼容两种方式）
- **安全警告** — 绝对不要将 API Key 硬编码在源码中（Issue #2）
- **leaderboard 格式化** — 非数值 metric 会导致格式化崩溃（Issue #6），建议在评估器中只返回数值指标
- **DataFilter 正常工作** — `core/data_filter.py` 过滤逻辑正确，Issue #3 已关闭
