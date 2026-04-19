# 目录结构规范

## 项目目录

```
model_eval/
├── .trae/                    # Trae IDE 规则配置
│   └── rules/
│       ├── dir-cofing.md     # 本文档
│       ├── development-specifications.md  # 开发规范
│       └── modelconfig.md    # 模型配置规范
├── cli/                      # 命令行入口
│   ├── __init__.py
│   ├── main.py               # CLI 主入口
│   └── local_main.py        # 本地评测入口
├── core/                     # 核心模块
│   ├── __init__.py
│   ├── auto_import.py       # 自动导入模块
│   ├── base.py               # 数据类定义（DataItem, ModelInput, ModelOutput, EvaluationResult）
│   ├── config.py             # 配置类（DatasetConfig, BenchmarkConfig, RunConfig）
│   ├── data_filter.py        # 数据过滤器（DataFilter）
│   ├── data_reader.py        # 数据读取器
│   ├── engine.py             # 评测引擎（EvaluationEngine）
│   ├── leaderboard.py        # 排行榜（Leaderboard）
│   └── registry.py          # 注册中心（Registry）
├── datasets/                  # 数据集模块
│   ├── __init__.py
│   ├── base.py               # 数据集基类（BaseDataset）
│   └── *.py                  # 具体数据集实现
├── evaluators/                # 评估器模块
│   ├── __init__.py
│   ├── base.py               # 评估器基类（BaseEvaluator）
│   └── *.py                  # 具体评估器实现
├── models/                    # 模型模块
│   ├── __init__.py
│   ├── base.py               # 模型基类（BaseModel）
│   └── *.py                  # 具体模型适配器
├── prompt_builder/            # 提示词构建器
│   ├── __init__.py
│   ├── base.py               # 提示词构建器基类（BasePromptBuilder）
│   └── *.py                  # 具体提示词构建器
├── reports/                   # 报告模块
│   ├── __init__.py
│   ├── base.py               # 报告基类（BaseReport）
│   └── formats.py            # 报告格式
├── tasks/                     # 任务运行器
│   ├── __init__.py
│   ├── task_runner.py        # 任务运行器基类（BaseTaskRunner）
│   ├── standard_runner.py     # 标准任务运行器（StandardTaskRunner）
│   └── multitask_runner.py    # 多任务运行器（MultiTaskRunner）
├── tools/                     # 工具模块
│   ├── __init__.py
│   ├── base.py               # 工具基类（BaseTool）
│   └── extensions.py         # 工具扩展
├── visualization/              # 可视化模块
│   ├── __init__.py
│   ├── base.py               # 可视化基类
│   └── platforms.py          # 可视化平台
├── configs/                   # 配置文件目录
│   └── test.yaml             # 测试配置
├── docs/                      # 文档目录
│   ├── architecture_design.md # 架构设计文档
│   └── api_reference.md      # API 参考文档
├── results/                   # 结果保存目录
│   └── run_YYYY-MM-DD_HH-MM-SS/
│       ├── config.json        # 评测配置
│       ├── results.jsonl      # 逐条评测结果
│       └── summary.json       # 汇总指标
├── tests/                     # 测试目录
│   └── *.py                  # 测试文件
├── .idea/                     # PyCharm 配置
├── .trae/                     # Trae IDE 配置
└── README.md                  # 项目说明
```

## 核心目录说明

| 目录 | 说明 |
|------|------|
| `core/` | 核心模块，包含数据类、引擎、注册中心等基础组件 |
| `datasets/` | 数据集模块，存放所有数据集实现 |
| `models/` | 模型模块，存放所有模型适配器 |
| `evaluators/` | 评估器模块，存放所有评估器 |
| `prompt_builder/` | 提示词构建器模块 |
| `tasks/` | 任务运行器模块，负责执行评测流程 |
| `reports/` | 报告生成模块 |
| `tools/` | 工具扩展模块 |
| `visualization/` | 可视化模块 |

## 模块命名规范

| 模块类型 | 目录 | 基类文件 | 实现文件命名 |
|----------|------|----------|--------------|
| 数据集 | `datasets/` | `base.py` | `数据集名.py`（如 `chinese_simpleqa.py`） |
| 模型 | `models/` | `base.py` | `模型名.py`（如 `test_minmax.py`） |
| 评估器 | `evaluators/` | `base.py` | `评估器名.py`（如 `accuracy.py`） |
| 提示词构建器 | `prompt_builder/` | `base.py` | `构建器名.py`（如 `qa_builder.py`） |
| 工具 | `tools/` | `base.py` | `工具名.py` |

## 结果目录结构

```
results/
└── run_{timestamp}/           # 每次运行生成独立的目录
    ├── config.json           # 评测配置副本
    ├── results.jsonl         # 逐条结果（JSONL格式）
    └── summary.json          # 汇总指标
```

## 配置文件目录

```
configs/
└── test.yaml                 # 测试用配置文件
```

## 添加新模块规范

### 添加新数据集

```
datasets/
├── __init__.py               # 自动导入，无需修改
├── base.py                   # 已存在
└── {数据集名}.py             # 新增数据集实现
```

### 添加新模型

```
models/
├── __init__.py               # 自动导入，无需修改
├── base.py                   # 已存在
└── {模型名}.py               # 新增模型适配器
```

### 添加新评估器

```
evaluators/
├── __init__.py               # 自动导入，无需修改
├── base.py                   # 已存在
└── {评估器名}.py            # 新增评估器实现
```
