---
name: dataset-researcher
description: 上网搜索数据集相关信息并下载数据集文件，整理成结构化的数据集分析报告。用于需要获取新评测数据集、下载数据文件、撰写数据集调研报告的场景。适用：搜索Hugging Face/GitHub上的数据集、下载数据文件、探索数据结构、生成数据报告。当用户提到"搜索数据集"、"下载数据集"、"调研数据集"、"找评测集"时使用此skill。
---

# 数据集搜索与调研 Skill

本 skill 负责在网络上搜索数据集相关信息、下载数据集文件、分析数据结构，并生成结构化的数据集分析报告。执行时应优先保证可复现性：记录来源、版本/提交号、下载文件、校验信息和分析命令，而不是只给出一个临时链接。

## 输入与交付物

开始前确认数据集名称、目标 split、是否允许下载完整数据，以及报告保存位置。默认将文件保存到 `data/{dataset_name}/`，将报告保存到 `dataset_report/{DatasetName}_Dataset_Report.md`；若用户指定路径，以用户指定路径为准。最终交付应包含实际文件路径、来源 URL、数据集版本或日期、文件大小（必要时 SHA-256）、数据格式、样本数、字段概览、质量检查结果和适配建议。

## 来源与可复现性要求

优先使用官方 Hugging Face/GitHub 仓库和论文页面；不要把搜索结果摘要当作数据集事实。记录 Hugging Face repo id、revision/commit、具体文件名和 split。下载前检查许可证、使用限制、隐私和是否需要认证；许可证不明确时在报告中明确标注"待确认"。网络失败时记录失败原因并尝试合规的备用官方来源，不要静默替换数据。

## 工作流程

### Step 1: 搜索数据集

使用 WebSearch 搜索目标数据集，收集以下信息：

| 需收集信息 | 搜索渠道 | 关键字段 |
|-----------|---------|---------|
| 数据集名称与简介 | WebSearch | 数据集全名、发布机构 |
| 数据文件位置 | Hugging Face | HF 数据集 ID / file 路径 |
| 官方仓库 | GitHub | README、论文、说明书 |
| 数据格式 | 仓库/README | jsonl/csv/parquet，字段定义 |

**推荐搜索关键词**：
```
<dataset_name> dataset benchmark       # 通用搜索
<dataset_name> huggingface datasets    # Hugging Face 搜索
"<dataset_name>" github paper          # 论文与仓库
```

**常用数据源**：
- **[Hugging Face Hub](https://huggingface.co/datasets)** — 首选，支持直接下载 parquet/jsonl/csv
- **[GitHub Releases](https://github.com)** — 大文件常用 Release 或 raw 链接
- **官方论文补充材料** — PDF 中可能包含数据链接
- **ModelScope** — 国内可用的阿里巴巴 HF 镜像

### Step 2: 下载数据集文件

先确认目标 split 和文件，再执行下载。默认使用 `skills/dataset-researcher/scripts/download_dataset.py`，它会自动保存到项目 `data/` 目录并生成同目录摘要；只有在需要特殊认证或文件类型时才手动调用库。下载后记录实际返回路径，不要假定文件名。

```bash
python skills/dataset-researcher/scripts/download_dataset.py <dataset_name> --hf <hf_repo_id> --file <filename>
# 或分析已有文件
python skills/dataset-researcher/scripts/download_dataset.py <dataset_name> --analyze-only --data-path <path>
```

大于 1GB 的数据集优先只下载目标 split 或小样本；下载后检查文件大小和（可行时）SHA-256。直接 URL 下载应使用可信官方地址，并避免覆盖同名文件。
```bash
# 方式1: huggingface_hub 下载（推荐）
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
# 下载单个文件
hf_hub_download(
    repo_id='<hf_repo_id>',
    filename='<filename>',
    repo_type='dataset',
    local_dir='data/<dataset_name>/'
)
# 或下载整个数据集
# snapshot_download(repo_id='<hf_repo_id>', repo_type='dataset', local_dir='data/<dataset_name>/')
"

# 方式2: 直接 HTTP 下载
curl -L -o data/<dataset_name>/<file> "<direct_url>"

# 方式3: 使用 datasets 库（如果安装了）
# from datasets import load_dataset
# ds = load_dataset('<hf_repo_id>', split='test')
# ds.to_json('data/<dataset_name>/data.jsonl')
```

### Step 3: 分析数据结构

读取并分析下载的数据文件，确认真实字段：

```python
import json
import pandas as pd

# 读取数据（根据扩展名选择）
if path.endswith(".jsonl"):
    data = [json.loads(l) for l in open(path, encoding="utf-8")]
elif path.endswith(".parquet"):
    data = pd.read_parquet(path).to_dict("records")
elif path.endswith(".csv"):
    data = pd.read_csv(path).to_dict("records")

print(f"总数据量: {len(data)}")
print(f"字段: {list(data[0].keys()) if data else 'empty'}")
print(f"示例: {json.dumps(data[0], ensure_ascii=False, indent=2)[:1000]}")
```

**必查字段**：
- **question/prompt/problem** — 问题描述
- **answer/reference/solution** — 参考答案
- **A/B/C/D** — 选项（多选题）
- **category/task/domain** — 分类标签
- **difficulty** — 难度等级

### Step 4: 生成数据集分析报告

按照 [dataset-analysis-report](../../CLAUDE.md) 的模板结构生成报告，保存到 `dataset_report/` 目录：

**报告结构**：
1. **简介**（来源、目标、应用场景、数据描述）
2. **数据集能力体系**（评估模型的什么通用能力）
3. **数据集场景体系**（数据分类体系）
4. **测评**（获取回复方法、测评方法、指标）

**报告文件名**：`{数据集名称}_Dataset_Report.md`

**报告内容必须基于下载的实际数据**，禁止假设字段。报告开头标注数据文件的实际存储路径。

### Step 5: 总结输出

输出最终结果，包括：
1. 数据集存储路径
2. 数据结构摘要
3. 报告保存位置
4. 下一步建议（适配到框架）

## 适配建议（报告末尾附录）

报告末尾附上适配建议，帮助后续使用 `dataset-adapter` skill：

```markdown
## 适配建议

- **适配器类型**: {标准问答 / 多选题 / 生成式评估 / 数学boxed}
- **推荐注册名**: `{DatasetName}`
- **推荐评估器**: `{accuracy / 自定义}`
- **字段映射**:
  - question → `DataItem.prompt`
  - answer → `DataItem.reference`
  - category → `DataItem.category`
- **配置示例**:
  ```yaml
  dataset:
    name: {DatasetName}
    params:
      data_path: "data/{dataset_name}/data.jsonl"
  ```
```

## 关键资源

| 资源 | 路径 | 用途 |
|------|------|------|
| 数据下载工具 | huggingface_hub / requests | 下载数据集 |
| 报告模板 | [dataset-analysis-report](../../CLAUDE.md) | 参考现有报告结构 |
| 已有报告 | `dataset_report/` | 参考报告格式 |
| 数据存储 | `data/` | 存放下载的数据文件 |
| 适配器模板 | `skills/dataset-adapter/SKILL.md` | 下一步适配指引 |

## 环境要求

### 依赖安装

```bash
pip install huggingface_hub  # 必须
# 可选
pip install datasets pandas pyarrow  # 大数据集处理
```

### 网络说明

- Hugging Face 国内访问可能较慢，如遇超时可尝试：
  - 配置代理：`export HF_ENDPOINT=https://hf-mirror.com`
  - 或使用 ModelScope 镜像

## 注意事项

- **必须先下载数据，才能写报告** — 数据字段以实际数据为准
- **报告中的数据结构以实际文件为准** — 不要假设字段名
- **记录下载失败情况** — 如果某个数据源失败，标注清楚并尝试备选源
- **控制数据量** — 如数据集过大（>1GB），可只下载子集做分析
- **检查数据质量** — 分析时要检查空值、异常值、格式不一致等问题
- **报告头标注路径** — 注明下载的数据存放在哪个目录，方便下一步适配
- **许可检查** — 记录数据集许可证，尤其是商业用途限制
