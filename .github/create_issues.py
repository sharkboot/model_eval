#!/usr/bin/env python3
"""
Batch create GitHub Issues script.
Usage: python .github/create_issues.py
"""
import subprocess
import json
import requests
import sys

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import subprocess
import json
import requests

# 从 git credential manager 动态获取 token（不硬编码）
result = subprocess.run(
    ["git", "credential", "fill"],
    input="protocol=https\nhost=github.com\n",
    capture_output=True, text=True, timeout=5
)
token = dict(line.split("=", 1) for line in result.stdout.strip().split("\n") if "=" in line).get("password", "")
if not token:
    print("ERROR: 无法获取 GitHub token，请先运行 `gh auth login`")
    exit(1)

REPO = "sharkboot/model_eval"
HEADERS = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
API = f"https://api.github.com/repos/{REPO}"


def post(path, data):
    r = requests.post(f"{API}/{path}", headers=HEADERS, json=data)
    return r.status_code, r.json()


def issue(title, body, labels=None):
    data = {"title": title, "body": body}
    if labels:
        data["labels"] = labels
    code, resp = post("issues", data)
    if code == 201:
        print(f"  ✓ #{resp['number']}: {title}")
    else:
        print(f"  ✗ {code}: {resp.get('message', resp)}")


print("=== 创建 Labels ===")
for lb in [
    ("critical", "B60205", "Critical bug or security issue"),
    ("security", "DA321F", "Security vulnerability or sensitive data exposure"),
    ("test", "0E8A16", "Test coverage or testing infrastructure"),
    ("refactor", "C5DEF5", "Code refactoring or restructuring"),
]:
    code, resp = post("labels", {"name": lb[0], "color": lb[1], "description": lb[2]})
    print(f"  {'✓' if code == 201 else '✗'} {lb[0]} ({code})")

print("\n=== 创建 Milestone ===")
code, resp = post("milestones", {
    "title": "v0.2.0 稳定性与质量改进",
    "description": "修复 critical bugs、增加测试覆盖、优化性能",
    "state": "open",
    "due_on": "2026-12-31",
})
if code == 201:
    print(f"  ✓ Milestone #{resp['number']} created")
    MILESTONE_NUM = resp["number"]
else:
    print(f"  ✗ {code}: {resp.get('message', resp)}")
    MILESTONE_NUM = None

print("\n=== 创建 Issues ===\n")

# Issue #1
issue(
    "[Bug] [Critical] core/base.py ModelOutput 类体内混入装饰器死代码",
    "## 问题描述\n\n"
    "**文件**: `core/base.py:10-12`\n\n"
    "```python\n"
    "# 当前错误代码（在 ModelOutput 类体内）\n"
    "class ModelOutput:\n"
    "    ...\n"
    "    def get_text(self) -> str:\n"
    "        ...\n"
    "        cls._registry.setdefault(group, {})[name] = obj  # ← 死代码！\n"
    "        return obj  # ← 死代码！\n"
    "    return wrapper  # ← 语法错误：在类方法体内\n"
    "```\n\n"
    "这三行代码显然是从 `@Registry.register` 装饰器里误粘贴进 `ModelOutput` 的，会导致 `SyntaxError` 或逻辑错误。\n\n"
    "## 预期行为\n\n"
    "`ModelOutput` 只包含 `get_text()` 和 `get_messages()` 两个方法，干净无多余代码。\n\n"
    "## 修复建议\n\n"
    "删除 `core/base.py` 中混入的这三行死代码，仅保留 `get_text()` 和 `get_messages()` 两个方法。",
    ["bug", "critical"]
)

# Issue #2
issue(
    "[Security] [Critical] models/test_minmax.py 硬编码 API Key 已泄露",
    "## 问题描述\n\n"
    "**文件**: `models/test_minmax.py:10`\n\n"
    "```python\n"
    "client = OpenAI(\n"
    '    api_key="sk-EiV60JloYYSdBbsh2SG0swtNTnlPxyKyI6ufMKOIFfziHx22",  # ← 泄漏！\n'
    '    base_url="https://api.ttxxvv.cn/v1"\n'
    ")\n"
    "```\n\n"
    "API Key 直接硬编码在源码中，已随 `git push` 上传到 GitHub 公共仓库，永久泄露。\n\n"
    "## 风险\n\n"
    "- 任何人可盗用该 Key 产生费用\n"
    "- 即使删除 commit，GitHub 历史中仍保留泄露记录\n\n"
    "## 修复建议\n\n"
    "1. **立即旋转 Key**：联系服务方重新生成 API Key，旧 Key 立即作废\n"
    "2. 将 key 改为从配置或环境变量读取：\n"
    "```python\n"
    "import os\n"
    'api_key = config.get("api_key") or os.environ.get("MINIMAX_API_KEY")\n'
    "client = OpenAI(api_key=api_key, base_url=...)\n"
    "```\n"
    "3. 将含 Key 的历史 commit 从仓库中彻底清除（`git filter-repo`）",
    ["bug", "critical", "security"]
)

# Issue #3
issue(
    "[Bug] [Critical] core/data_filter.py apply() 方法语法错误，过滤逻辑不生效",
    "## 问题描述\n\n"
    "**文件**: `core/data_filter.py:10-14`\n\n"
    "```python\n"
    "def apply(self, data_items: List[Any]) -> List[Any]:\n"
    "    result = data_items\n"
    "    ...\n"
    "    return result   # ← 先返回，后续代码永远不执行\n"
    '    return f"Unsupported file format: {ext}"  # ← 死代码，ext 未定义\n'
    "```\n\n"
    "方法在中间位置提前 `return result`，后面一行 `return f\"Unsupported...\"` 永远不会执行，且 `{ext}` 变量未定义会引发 `NameError`。\n\n"
    "## 预期行为\n\n"
    "根据 `categories_include` / `categories_exclude` / `custom_filter` 正确过滤数据项。\n\n"
    "## 修复建议\n\n"
    "删除最后一行错误代码，修复后应为：\n"
    "```python\n"
    "def apply(self, data_items: List[Any]) -> List[Any]:\n"
    "    result = data_items\n"
    "    if self.categories_include:\n"
    "        result = [item for item in result if any(cat in item.category for cat in self.categories_include)]\n"
    "    if self.categories_exclude:\n"
    "        result = [item for item in result if not any(cat in item.category for cat in self.categories_exclude)]\n"
    "    if self.custom_filter:\n"
    "        result = [item for item in result if self.custom_filter(item)]\n"
    "    return result\n"
    "```",
    ["bug", "critical"]
)

# Issue #4
issue(
    "[Bug] [High] evaluators/base.py 评估器签名不一致且返回类型错误",
    "## 问题描述\n\n"
    "**文件**: `evaluators/base.py`\n\n"
    "**问题 A** — 基类签名有多余逗号：\n"
    "```python\n"
    "def evaluate(self, pred: str, data_item: DataItem, ) -> EvaluationResult:  # ← 多余逗号\n"
    "```\n\n"
    "**问题 B** — 实现类签名与基类不一致，且返回 `dict` 而非 `EvaluationResult`：\n"
    "```python\n"
    "class AccuracyEvaluator(BaseEvaluator):\n"
    "    def evaluate(self, pred: str, item: DataItem):   # ← 参数名不同\n"
    '        acc = 1.0 if pred.strip() == str(item.reference).strip() else 0.0\n'
    '        return {"accuracy": acc}  # ← 应为 EvaluationResult 对象\n'
    "```\n\n"
    "## 预期行为\n\n"
    "所有评估器统一继承 `BaseEvaluator`，签名一致，返回 `EvaluationResult`。\n\n"
    "## 修复建议\n\n"
    "```python\n"
    '@Registry.register(\"accuracy\", \"evaluator\")\n'
    "class AccuracyEvaluator(BaseEvaluator):\n"
    "    def evaluate(self, pred: str, data_item: DataItem) -> EvaluationResult:\n"
    '        acc = 1.0 if pred.strip() == str(data_item.reference).strip() else 0.0\n'
    "        return EvaluationResult(\n"
    '            data_id=data_item.id,\n'
    '            evaluator_name=\"accuracy\",\n'
    '            raw_output=acc,\n'
    '            metrics={\"accuracy\": acc},\n'
    "        )\n"
    "```",
    ["bug"]
)

# Issue #5
issue(
    "[Perf] [High] StandardTaskRunner 并发写入无缓冲，崩溃时数据可能丢失",
    "## 问题描述\n\n"
    "**文件**: `tasks/standard_runner.py:133-136`\n\n"
    "```python\n"
    "def _append_record(self, record):\n"
    '    with self._lock:\n'
    '        with open(self.result_file, \"a\", encoding=\"utf-8\") as f:\n'
    '            f.write(json.dumps(record, ensure_ascii=False) + \"\\n\")\n'
    "```\n\n"
    "每次写入都单独打开/关闭文件，高并发下产生大量 I/O。更严重的是：如果进程在写入中途崩溃，可能产生不完整行，导致后续加载出错。\n\n"
    "## 修复建议\n\n"
    "1. 改为批量缓冲写入（积攒 N 条后 flush）\n"
    "2. 写入临时文件（`.tmp`），完成原子重命名（`os.replace`）\n"
    "3. 加载时校验每行 JSON 完整性",
    ["enhancement", "refactor"]
)

# Issue #6
issue(
    "[Bug] [High] core/leaderboard.py 非数值 metric 导致格式化 TypeError",
    "## 问题描述\n\n"
    "**文件**: `core/leaderboard.py:23-24`\n\n"
    "```python\n"
    'for k, v in metrics.items():\n'
    '    logger.info(f\"  {k}: {v:.4f}\")  # ← v 若不是 float/int 则 TypeError\n'
    "```\n\n"
    "若 evaluator 返回字符串类指标（如分类结果、布尔值），格式化会抛 `TypeError`。\n\n"
    "## 修复建议\n\n"
    "```python\n"
    'for k, v in metrics.items():\n'
    "    if isinstance(v, (int, float)):\n"
    '        logger.info(f\"  {k}: {v:.4f}\")\n'
    "    else:\n"
    '        logger.info(f\"  {k}: {v}\")\n'
    "```",
    ["bug"]
)

# Issue #7
issue(
    "[UX] [Medium] EvaluationEngine.run() 阻塞等待回车，无法 CI 集成",
    "## 问题描述\n\n"
    "**文件**: `core/engine.py:71-72`\n\n"
    "```python\n"
    'if self.visualizer:\n'
    '    input(\"Press Enter to stop visualization...\")  # ← 阻塞主线程\n'
    "```\n\n"
    "可视化启动后强制等待用户按回车，无法在 CI/CD 或非交互环境下运行。\n\n"
    "## 修复建议\n\n"
    "- 添加配置项 `auto_exit: true` 跳过等待\n"
    "- 支持 `SIGINT` / `SIGTERM` 信号优雅退出\n"
    "- 生产模式（`--no-visualize` 或环境变量 `CI=true`）自动禁用阻塞",
    ["enhancement"]
)

# Issue #8
issue(
    "[Test] [Medium] 测试目录为空，核心模块无单元测试",
    "## 问题描述\n\n"
    "`tests/` 目录下只有 `__init__.py`，无任何测试文件。核心组件（`DataFilter`、`StandardTaskRunner`、`Leaderboard`、`Registry`）均无测试覆盖。\n\n"
    "## 修复建议\n\n"
    "引入 `pytest`，优先为以下模块补充测试：\n"
    "1. `core/data_filter.py` — DataFilter.apply() 三种过滤模式\n"
    "2. `core/registry.py` — 注册/获取/列表\n"
    "3. `evaluators/base.py` — AccuracyEvaluator\n"
    "4. `core/leaderboard.py` — pretty_print 不崩溃\n\n"
    "在 `pyproject.toml` 或 `setup.cfg` 中配置 pytest，并在 CI 中运行 `pytest tests/`。",
    ["test"]
)

# Issue #9
issue(
    "[Bug] [Medium] core/auto_import.py 导入失败无日志，调试困难",
    "## 问题描述\n\n"
    "**文件**: `core/auto_import.py`\n\n"
    "```python\n"
    "def auto_import(package_name):\n"
    "    package = importlib.import_module(package_name)\n"
    '    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + \".\"):\n'
    "        importlib.import_module(module_name)  # ← 任何 ImportError 导致整个流程中断，无日志\n"
    "```\n\n"
    "某个 adapter 模块导入失败会直接抛异常，且没有任何日志提示是哪个文件出了问题。\n\n"
    "## 修复建议\n\n"
    "```python\n"
    "def auto_import(package_name, verbose=False):\n"
    "    package = importlib.import_module(package_name)\n"
    '    for _, module_name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + \".\"):\n'
    "        try:\n"
    "            importlib.import_module(module_name)\n"
    "        except ImportError as e:\n"
    '            logger.warning(f\"Failed to import {module_name}: {e}\")\n'
    "            if verbose:\n"
    "                raise\n"
    "```",
    ["bug"]
)

# Issue #10
issue(
    "[Doc] [Low] 配置字段名不统一：path vs data_path",
    "## 问题描述\n\n"
    "| 位置 | 字段名 |\n"
    "|------|--------|\n"
    "| `configs/test.yaml:8` | `data_path: E:/eval/...` |\n"
    "| `README.md:83` | `path: \"data/chinese_simpleqa.json\"` |\n"
    "| `adapter/chinese_simpleqa/chinese_simpleqa.py:26` | 读取 `config.get('data_path')` |\n\n"
    "README 示例文档使用 `path`，但代码实际读取 `data_path`，用户按文档配置会报 `ValueError: requires 'data_path' config`。\n\n"
    "## 修复建议\n\n"
    "统一字段名为 `data_path`，并更新 README 示例代码。",
    ["documentation"]
)

print("\n✅ 全部完成！")
