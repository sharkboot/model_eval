├── docs/
│   ├── architecture_design.md      # 本文档
│   ├── api_reference.md            # API 参考文档
│   └── user_guide.md               # 用户使用指南
├── core/
│   ├── __init__.py
│   ├── base.py                     # 基础类定义（DataItem, ModelInput等）
│   ├── config.py                   # 配置加载器
│   ├── engine.py                   # 评测引擎
│   ├── registry.py                 # 注册中心基类
│   └── analyzer.py                 # 数据分析器
├── datasets/
│   ├── __init__.py
│   ├── base.py                     # BaseDataset
│   ├── registry.py                 # DatasetRegistry
│   ├── mmlu.py                     # MMLU 数据集
│   ├── gsm8k.py                    # GSM8K 数据集
│   └── custom.py                   # 自定义数据集模板
├── models/
│   ├── __init__.py
│   ├── base.py                     # BaseModel
│   ├── registry.py                 # ModelRegistry
│   ├── openai.py                    # OpenAI 模型
│   └── anthropic.py                # Anthropic 模型
├── evaluators/
│   ├── __init__.py
│   ├── base.py                     # BaseEvaluator
│   ├── registry.py                 # EvaluatorRegistry
│   ├── exact_match.py              # 精确匹配评估器
│   ├── rouge.py                    # Rouge 评分评估器
│   ├── llm_judge.py                # LLM 评判评估器
│   └── composite.py                # 组合评估器
├── examples/
│   ├── config/
│   │   ├── mmlu_eval.json          # MMLU 评测配置
│   │   └── gsm8k_eval.json         # GSM8K 评测配置
│   └── scripts/
│       └── run_evaluation.py       # 评测脚本示例
├── tests/
│   ├── test_dataset.py             # 数据集测试
│   ├── test_evaluator.py           # 评估器测试
│   └── test_engine.py              # 引擎测试
├── requirements.txt                 # 依赖列表
└── README.md                       # 项目说明