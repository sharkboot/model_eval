# Model Evaluation Framework

## 项目简介

本项目是一个参考EvalScope实现的模型评测框架，支持大语言模型（LLM）、多模态模型（VLM）、嵌入模型（Embedding）、重排模型（Reranker）和生成模型（AIGC）的评测。框架提供了多后端评估、性能监控（压测）和工具扩展等核心功能，支持多种输出格式和可视化平台。

## 项目结构

```
model_eval/
├── models/            # 模型模块，包括API模型和本地模型
├── datasets/          # 数据集模块，包括标准评测基准和自定义数据集
├── backends/          # 后端评估模块，包括原生后端和第三方框架集成
├── performance/       # 性能监控模块，支持并发测试和指标追踪
├── tools/             # 工具扩展模块，集成Tool-Bench、Needle-in-a-Haystack等工具
├── reports/           # 报告生成模块，支持JSON、表格和日志等格式
├── visualization/     # 可视化平台模块，支持Gradio、Wandb、SwanLab、ClearML等
├── configs/           # 配置文件目录
├── examples/          # 使用示例目录
├── utils/             # 工具函数目录
└── README.md          # 项目说明文件
```

## 安装说明

### 基本依赖

本框架不依赖任何外部库，使用纯Python实现。

### 可选依赖

根据需要安装以下依赖：

- **可视化平台**：
  ```bash
  pip install gradio
  ```

## 使用方法

### 1. 配置文件

创建配置文件（参考 `configs/example_config.json`），指定模型、数据集、后端、性能测试和报告等配置。

### 2. 运行评测

使用评测框架执行脚本运行评测，默认使用 `configs/example_config.json` 配置文件：

```bash
python run_eval.py
```

使用指定的配置文件运行评测：

```bash
python run_eval.py --config configs/test_config.json
```

### 3. 查看结果

评测结果将保存在 `results/` 目录中，包括：
- `eval_results.json`：评测结果
- `performance_metrics.json`：性能测试指标

## 示例

### 评估OpenAI模型

```python
from models import OpenAIModel
from datasets import MMLUDataset
from backends import NativeBackend
from reports import JSONReport

# 初始化模型
model = OpenAIModel({
    "api_key": "YOUR_API_KEY",
    "model_name": "gpt-3.5-turbo",
    "base_url": "https://api.openai.com/v1"
})

# 初始化数据集
dataset = MMLUDataset({"split": "test"})

# 初始化后端
backend = NativeBackend({"task_type": "llm"})

# 运行评测
results = backend.evaluate(model, dataset)

# 生成报告
report = JSONReport({})
for result in results:
    report.add_result(result)
report.save("results/eval_results.json")
```

### 性能测试

```python
from performance import ConcurrencyTest

# 初始化性能测试
perf_test = ConcurrencyTest({
    "concurrency": 5,
    "requests": 100
})

# 设置模型和数据集
perf_test.setup(model, dataset)

# 运行性能测试
metrics = perf_test.run()
print(metrics)
```

### 可视化结果

```python
from visualization import WandbVisualizer

# 初始化可视化工具
viz = WandbVisualizer({
    "project": "model_eval",
    "name": "eval_run"
})

# 可视化结果
viz.visualize(metrics)
viz.save()
```

## 配置说明

配置文件采用JSON格式，包含以下部分：

### 模型配置

```json
"model": {
  "type": "api",
  "name": "OpenAIModel",
  "config": {
    "api_key": "YOUR_API_KEY",
    "model_name": "gpt-3.5-turbo",
    "base_url": "https://api.openai.com/v1"
  }
}
```

### 数据集配置

#### 标准数据集

```json
"dataset": {
  "type": "standard",
  "name": "MMLUDataset",
  "config": {
    "split": "test"
  }
}
```

#### 自定义数据集

支持多种文件格式：JSON、JSONL、CSV、XLSX、Parquet

```json
"dataset": {
  "type": "custom",
  "name": "MCQDataset",
  "config": {
    "data_path": "data/mcq_sample.json",
    "split": "test"
  }
}
```

### 后端配置

```json
"backend": {
  "type": "native",
  "name": "NativeBackend",
  "config": {
    "task_type": "llm"
  }
}
```

### 性能测试配置

```json
"performance": {
  "type": "concurrency",
  "name": "ConcurrencyTest",
  "config": {
    "concurrency": 5,
    "requests": 100
  }
}
```

### 报告配置

```json
"report": {
  "type": "json",
  "name": "JSONReport",
  "config": {
    "output_path": "results/eval_results.json"
  }
}
```

### 可视化配置

```json
"visualization": {
  "type": "wandb",
  "name": "WandbVisualizer",
  "config": {
    "project": "model_eval",
    "name": "eval_run"
  }
}
```

## 扩展指南

### 添加新模型

1. 在 `models/` 目录下创建新的模型类，继承自 `BaseModel`
2. 实现 `generate` 和 `get_model_info` 方法
3. 在 `models/__init__.py` 中导出新模型类

### 添加新数据集

1. 在 `datasets/` 目录下创建新的数据集类，继承自 `BaseDataset`
2. 实现 `load` 和 `get_dataset_info` 方法
3. 在 `datasets/__init__.py` 中导出新数据集类

### 添加新后端

1. 在 `backends/` 目录下创建新的后端类，继承自 `BaseBackend`
2. 实现 `evaluate` 和 `get_backend_info` 方法
3. 在 `backends/__init__.py` 中导出新后端类

### 添加新工具

1. 在 `tools/` 目录下创建新的工具类，继承自 `BaseTool`
2. 实现 `setup`、`run` 和 `get_results` 方法
3. 在 `tools/__init__.py` 中导出新工具类

## 许可证

本项目采用 MIT 许可证。
