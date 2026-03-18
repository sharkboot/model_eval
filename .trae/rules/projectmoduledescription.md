### 一、输入层

&#x20;

作为评测的数据源入口，涵盖模型与数据集两大维度：

1. **模型来源**：
   - API 模型（如 OpenAI API、Anthropic Claude API 等）；
   - 本地模型（基于 ModelScope 加载的本地化部署模型）。
2. **数据集**：
   - 标准评测基准（如 MMLU、GSM8K、C-Eval、HumanEval 等通用 / 领域专用基准）；
   - 自定义数据（如 MCQ 选择题、QA 问答对、自定义 Function-Call 数据集、多模态 VQA 数据等）。

### 二、核心功能层

框架的核心能力集合，支撑多维度、多场景的评测需求：

1. **多后端评估**：
   - 原生后端：统一支持 LLM、VLM、Embedding、Reranker、AIGC 等模型的原生评测；
   - 集成第三方框架：无缝对接 OpenCompass、MTEB、VLMEvalKit、RAGAS 等专业评测后端，覆盖不同类型模型 / 任务的评测需求。
2. **性能监控（压测）**：
   - 模型插件：适配多种模型服务 API 协议，支持 LLM/VLM/Embedding 等模型服务的压力测试；
   - 数据插件：兼容多类输入数据格式；
   - 指标追踪：量化 TTFT、TPOT、吞吐量、稳定性等核心性能指标，还支持 SLA 自动调优（测试特定延迟 / 吞吐量下的最大并发）。
3. **工具扩展**：
   - 集成 Tool-Bench、Needle-in-a-Haystack、BFCL-v3/τ-bench 等工具 / 基准，支持 AI Agent、RAG 等复杂场景的评测。

### 三、输出层

评测结果的呈现与交付形式，兼顾结构化与可视化：

1. **结构化报告**：支持 JSON、表格（Table）、日志（Logs）等机器可读 / 人工可读的结构化输出；
2. **可视化平台**：对接 Gradio（WebUI 交互界面）、Wandb、SwanLab、ClearML 等工具，支持多模型对比、报告概览 / 详情查看、压测指标可视化等。

