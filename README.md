# Model Evaluation Framework

## 项目简介

本项目是一个轻量级的大语言模型评测框架，支持自定义数据集、模型适配器、评估器和提示词构建器的灵活扩展。框架采用注册中心模式实现模块解耦，支持并发评测和断点续跑。

## 项目结构

```
model_eval/
├── cli/                  # 命令行入口
│   ├── main.py           # CLI 主入口
│   └── local_main.py     # 本地评测入口
├── core/                 # 核心模块
│   ├── auto_import.py    # 自动导入
│   ├── base.py           # 数据类定义
│   ├── config.py         # 配置类
│   ├── data_filter.py    # 数据过滤器
│   ├── data_reader.py    # 数据读取器
│   ├── engine.py         # 评测引擎
│   ├── leaderboard.py    # 排行榜
│   └── registry.py       # 注册中心
├── datasets/             # 数据集模块
│   ├── base.py           # 数据集基类
│   └── *.py              # 具体数据集实现
├── evaluators/           # 评估器模块
│   ├── base.py           # 评估器基类
│   └── *.py              # 具体评估器实现
├── models/               # 模型模块
│   ├── base.py           # 模型基类
│   └── *.py              # 具体模型适配器
├── prompt_builder/        # 提示词构建器
│   ├── base.py           # 提示词构建器基类
│   └── *.py              # 具体提示词构建器
├── reports/              # 报告模块
│   ├── base.py           # 报告基类
│   └── formats.py        # 报告格式
├── tasks/                # 任务运行器
│   ├── task_runner.py    # 任务运行器基类
│   ├── standard_runner.py # 标准任务运行器
│   └── multitask_runner.py # 多任务运行器
├── tools/                # 工具模块
│   ├── base.py           # 工具基类
│   └── extensions.py     # 工具扩展
├── visualization/         # 可视化模块
│   ├── base.py           # 可视化基类
│   └── platforms.py      # 可视化平台
├── configs/              # 配置文件目录
├── docs/                 # 文档目录
├── results/              # 结果保存目录
└── tests/                # 测试目录
```

## 安装说明

### 基本依赖

```bash
pip install openai pyyaml
```

### 可选依赖

```bash
# 数据可视化
pip install matplotlib seaborn

# 其他工具
pip install pandas openpyxl
```

## 快速开始

### 1. 配置文件

创建评测配置文件（参考 `configs/test.yaml`）：

```yaml
# 数据集配置
dataset:
  name: ChineseSimpleQA
  params:
    path: "data/chinese_simpleqa.json"

# 模型配置
model:
  name: MiniMax
  params:
    generation_config:
      temperature: 0.7
      max_tokens: 1024

# 评估器配置
evaluators:
  - name: accuracy

# 提示词构建器配置
prompt_builder:
  name: qa_builder
  params: {}

# 并发配置
num_workers: 4

# 输出配置
output_path: results
run_name: my_eval
```

### 2. 运行评测

```bash
# 使用CLI运行
python -m cli.main --config configs/test.yaml

# 或直接运行
python cli/main.py --config configs/test.yaml
```

### 3. 查看结果

评测结果保存在 `results/{run_name}/` 目录中：

```
results/my_eval/
├── config.json      # 评测配置
├── results.jsonl    # 逐条评测结果
└── summary.json     # 汇总指标
```

## 核心概念

### 数据类

框架定义了三个核心数据类：

| 类名 | 说明 |
|------|------|
| `DataItem` | 标准数据单元，包含 id、prompt、reference、metadata 等字段 |
| `ModelInput` | 模型输入，支持 text 和 chat 两种模式 |
| `ModelOutput` | 模型输出，包含生成文本和使用统计 |

### 注册中心

采用分组注册机制，支持四种模块类型：

| group | 说明 | 装饰器 |
|-------|------|--------|
| dataset | 数据集 | `@Registry.register("name", "dataset")` |
| model | 模型 | `@Registry.register("name", "model")` |
| evaluator | 评估器 | `@Registry.register("name", "evaluator")` |
| prompt_builder | 提示词构建器 | `@Registry.register("name", "prompt_builder")` |

## 扩展指南

### 添加新数据集

```python
from datasets.base import BaseDataset
from core.base import DataItem
from core.registry import Registry

@Registry.register("MyDataset", "dataset")
class MyDataset(BaseDataset):
    def load_raw_data(self) -> List[Dict]:
        # 实现数据加载逻辑
        return []

    def preprocess(self, data_item: Dict) -> DataItem:
        return DataItem(
            id=self.build_id(data_item.get("id")),
            prompt=data_item["question"],
            reference=data_item["answer"],
            metadata=data_item.get("metadata", {})
        )
```

### 添加新模型

```python
from models.base import BaseModel
from core.base import ModelInput, ModelOutput
from core.registry import Registry

@Registry.register("MyModel", "model")
class MyModel(BaseModel):
    def generate(self, model_input: ModelInput) -> ModelOutput:
        # 调用模型 API
        response = your_model_api(model_input.prompt)
        return ModelOutput(
            type="text",
            text=response.text,
            usage={"tokens": response.usage.total_tokens}
        )
```

### 添加新评估器

```python
from evaluators.base import BaseEvaluator
from core.base import DataItem
from core.registry import Registry

@Registry.register("my_accuracy", "evaluator")
class MyAccuracyEvaluator(BaseEvaluator):
    def evaluate(self, pred: str, data_item: DataItem):
        correct = pred.strip().lower() == str(data_item.reference).strip().lower()
        return {"accuracy": 1.0 if correct else 0.0}
```

### 添加新提示词构建器

```python
from prompt_builder.base import BasePromptBuilder
from core.base import ModelInput
from core.registry import Registry

@Registry.register("my_builder", "prompt_builder")
class MyPromptBuilder(BasePromptBuilder):
    def build(self, item):
        return ModelInput(
            type="text",
            prompt=f"请根据以下问题回答：{item.prompt}\n\n请用JSON格式输出答案。"
        )
```

## 评测流程

```
配置文件 (YAML)
    ↓
EvaluationEngine
    ↓
自动导入模块 → Registry 注册
    ↓
创建 StandardTaskRunner
    ↓
┌─────────────────────────────────────┐
│  循环处理每个数据项                    │
│                                      │
│  DataItem → PromptBuilder → ModelInput │
│       ↓                              │
│  Model.generate() → ModelOutput      │
│       ↓                              │
│  Evaluator.evaluate() → Metrics      │
│       ↓                              │
│  写入 results.jsonl                  │
└─────────────────────────────────────┘
    ↓
汇总指标 → summary.json
```

## 配置说明

### 数据集配置

```yaml
dataset:
  name: ChineseSimpleQA
  params:
    path: "data/dataset.json"  # 数据路径
    limits: 100                # 限制数据量（可选）
```

### 模型配置

```yaml
model:
  name: MiniMax
  params:
    generation_config:         # 生成配置
      temperature: 0.7
      max_tokens: 1024
      top_p: 0.9
```

### 评估器配置

```yaml
evaluators:
  - name: accuracy              # 精确匹配
  - name: rouge                 # ROUGE 评分（需实现）
```

### 提示词构建器配置

```yaml
prompt_builder:
  name: qa_builder
  params:
    system_prompt: "你是一个有帮助的AI助手。"  # 可选
```

### 并发配置

```yaml
num_workers: 4    # 并发线程数，默认1
```

### 输出配置

```yaml
output_path: results     # 输出目录
run_name: my_experiment  # 运行名称（可选，自动生成时间戳）
```

## 示例

### 基本评测

```python
from core.engine import EvaluationEngine

config = {
    "dataset": {
        "name": "ChineseSimpleQA",
        "params": {"path": "data/test.json"}
    },
    "model": {
        "name": "MiniMax",
        "params": {"generation_config": {"temperature": 0.7}}
    },
    "evaluators": [
        {"name": "accuracy"}
    ],
    "prompt_builder": {
        "name": "qa_builder",
        "params": {}
    },
    "num_workers": 4
}

engine = EvaluationEngine(config)
results = engine.run()
```

### 自定义评测流程

```python
from core.registry import Registry
from datasets.base import BaseDataset
from models.base import BaseModel
from evaluators.base import BaseEvaluator

# 注册组件
@Registry.register("my_dataset", "dataset")
class MyDataset(BaseDataset): ...

@Registry.register("my_model", "model")
class MyModel(BaseModel): ...

@Registry.register("my_evaluator", "evaluator")
class MyEvaluator(BaseEvaluator): ...

# 使用
dataset = Registry.create("my_dataset", "dataset", path="data.json")
model = Registry.create("my_model", "model", api_key="xxx")
evaluator = Registry.create("my_evaluator", "evaluator", threshold=0.5)

# 运行评测
data = dataset.load()
for item in data:
    model_input = ...
    output = model.generate(model_input)
    result = evaluator.evaluate(output.get_text(), item)
```

## 许可证

本项目采用 MIT 许可证。
