# GitHub Issue 批量创建脚本
# 使用方法：在 E:/LLM/model_eval 目录下运行
#   powershell -ExecutionPolicy Bypass -File .github/create_issues.ps1

$GH = "C:\Program Files\GitHub CLI\gh.exe"
$REPO = "sharkboot/model_eval"

# 确保 gh 已登录
& $GH auth status | Out-Null

Write-Host "=== 创建 Labels ===" -ForegroundColor Cyan

& $GH label create critical --color B60205 --description "Critical bug or security issue requiring immediate attention" --repo $REPO 2>$null
& $GH label create security --color DA321F --description "Security vulnerability or sensitive data exposure" --repo $REPO 2>$null
& $GH label create test --color 0E8A16 --description "Test coverage or testing infrastructure" --repo $REPO 2>$null
& $GH label create refactor --color C5DEF5 --description "Code refactoring or restructuring" --repo $REPO 2>$null

Write-Host "=== 创建 Milestone ===" -ForegroundColor Cyan
& $GH api "repos/$REPO/milestones" -f title="v0.2.0 稳定性与质量改进" -f description="修复 critical bugs、增加测试覆盖、优化性能" -f state="open" -f due_on="2026-12-31" --method POST 2>$null | ForEach-Object { if ($_ -ne "null") { Write-Host "Milestone created" } }

Write-Host ""
Write-Host "=== 创建 Issues ===" -ForegroundColor Cyan

# Issue #1
& $GH issue create --repo $REPO --title "[Bug] [Critical] core/base.py ModelOutput 类体内混入装饰器死代码" `
  --label "bug,critical" `
  --body @"
## 问题描述

**文件**: \`core/base.py:10-12\`

\`\`\`python
# 当前错误代码（在 ModelOutput 类体内）
class ModelOutput:
    ...
    def get_text(self) -> str:
        ...
        cls._registry.setdefault(group, {})[name] = obj  # ← 死代码！
        return obj  # ← 死代码！
    return wrapper  # ← 语法错误：在类方法体内
\`\`\`

这三行代码显然是从 \`@Registry.register\` 装饰器里误粘贴进 \`ModelOutput\` 的，会导致 \`SyntaxError\` 或逻辑错误。

## 预期行为

\`ModelOutput\` 只包含 \`get_text()\` 和 \`get_messages()\` 两个方法，干净无多余代码。

## 修复建议

删除\`core/base.py\`中混入的这三行死代码：
\`\`\`python
        cls._registry.setdefault(group, {})[name] = obj
        return obj
    return wrapper
\`\`\`
仅保留 \`get_text()\` 和 \`get_messages()\` 两个方法。
"@

# Issue #2
& $GH issue create --repo $REPO --title "[Security] [Critical] models/test_minmax.py 硬编码 API Key 已泄露" `
  --label "bug,critical,security" `
  --body @"
## 问题描述

**文件**: \`models/test_minmax.py:10\`

\`\`\`python
client = OpenAI(
    api_key=\"sk-EiV60JloYYSdBbsh2SG0swtNTnlPxyKyI6ufMKOIFfziHx22\",  # ← 泄漏！
    base_url=\"https://api.ttxxvv.cn/v1\"
)
\`\`\`

API Key 直接硬编码在源码中，已随 \`git push\` 上传到 GitHub 公共仓库，永久泄露。

## 风险

- 任何人可盗用该 Key 产生费用
- 即使删除 commit，GitHub 历史中仍保留泄露记录

## 修复建议

1. **立即旋转 Key**：联系服务方重新生成 API Key，旧 Key 立即作废
2. 将 key 改为从配置或环境变量读取：
\`\`\`python
import os
api_key = config.get(\"api_key\") or os.environ.get(\"MINIMAX_API_KEY\")
client = OpenAI(api_key=api_key, base_url=...)
\`\`\`
3. 将含 Key 的历史 commit 从仓库中彻底清除（\`git filter-repo\` ）
"@

# Issue #3
& $GH issue create --repo $REPO --title "[Bug] [Critical] core/data_filter.py apply() 方法语法错误，过滤逻辑不生效" `
  --label "bug,critical" `
  --body @"
## 问题描述

**文件**: \`core/data_filter.py:10-14\`

\`\`\`python
def apply(self, data_items: List[Any]) -> List[Any]:
    result = data_items
    ...
    return result   # ← 先返回，后续代码永远不执行
    return f\"Unsupported file format: {ext}\"  # ← 死代码，ext 未定义
\`\`\`

方法在中间位置提前 \`return result\`，后面一行 \`return f\"Unsupported...\"\` 永远不会执行，且 \`{ext}\` 变量未定义会引发 \`NameError\`。

## 预期行为

根据 \`categories_include\` / \`categories_exclude\` / \`custom_filter\` 正确过滤数据项。

## 修复建议

删除最后一行错误代码，修复后应为：
\`\`\`python
def apply(self, data_items: List[Any]) -> List[Any]:
    result = data_items
    if self.categories_include:
        result = [item for item in result if any(cat in item.category for cat in self.categories_include)]
    if self.categories_exclude:
        result = [item for item in result if not any(cat in item.category for cat in self.categories_exclude)]
    if self.custom_filter:
        result = [item for item in result if self.custom_filter(item)]
    return result
\`\`\`
"@

# Issue #4
& $GH issue create --repo $REPO --title "[Bug] [High] evaluators/base.py 评估器签名不一致且返回类型错误" `
  --label "bug" `
  --body @"
## 问题描述

**文件**: \`evaluators/base.py\`

**问题 A** — 基类签名有多余逗号：
\`\`\`python
def evaluate(self, pred: str, data_item: DataItem, ) -> EvaluationResult:  # ← 多余逗号
\`\`\`

**问题 B** — 实现类签名与基类不一致，且返回 \`dict\` 而非 \`EvaluationResult\`：
\`\`\`python
class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, pred: str, item: DataItem):   # ← 参数名不同
        acc = 1.0 if pred.strip() == str(item.reference).strip() else 0.0
        return {\"accuracy\": acc}  # ← 应为 EvaluationResult 对象
\`\`\`

## 预期行为

所有评估器统一继承 \`BaseEvaluator\`，签名一致，返回 \`EvaluationResult\`。

## 修复建议

\`\`\`python
@Registry.register(\"accuracy\", \"evaluator\")
class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, pred: str, data_item: DataItem) -> EvaluationResult:
        acc = 1.0 if pred.strip() == str(data_item.reference).strip() else 0.0
        return EvaluationResult(
            data_id=data_item.id,
            evaluator_name=\"accuracy\",
            raw_output=acc,
            metrics={\"accuracy\": acc},
        )
\`\`\`
"@

# Issue #5
& $GH issue create --repo $REPO --title "[Perf] [High] StandardTaskRunner 并发写入无缓冲，崩溃时数据可能丢失" `
  --label "enhancement,refactor" `
  --body @"
## 问题描述

**文件**: \`tasks/standard_runner.py:133-136\`

\`\`\`python
def _append_record(self, record):
    with self._lock:
        with open(self.result_file, \"a\", encoding=\"utf-8\") as f:
            f.write(json.dumps(record, ensure_ascii=False) + \"\\n\")
\`\`\`

每次写入都单独打开/关闭文件，高并发下产生大量 I/O。更严重的是：如果进程在写入中途崩溃，可能产生不完整行，导致后续加载出错。

## 修复建议

1. 改为批量缓冲写入（积攒 N 条后 flush）
2. 写入临时文件（\`.tmp\`），完成原子重命名（\`os.replace\`）
3. 加载时校验每行 JSON 完整性
"@

# Issue #6
& $GH issue create --repo $REPO --title "[Bug] [High] core/leaderboard.py 非数值 metric 导致格式化 TypeError" `
  --label "bug" `
  --body @"
## 问题描述

**文件**: \`core/leaderboard.py:23-24\`

\`\`\`python
for k, v in metrics.items():
    logger.info(f\"  {k}: {v:.4f}\")  # ← v 若不是 float/int 则 TypeError
\`\`\`

若 evaluator 返回字符串类指标（如分类结果、布尔值），格式化会抛 \`TypeError\`。

## 修复建议

\`\`\`python
for k, v in metrics.items():
    if isinstance(v, (int, float)):
        logger.info(f\"  {k}: {v:.4f}\")
    else:
        logger.info(f\"  {k}: {v}\")
\`\`\`
"@

# Issue #7
& $GH issue create --repo $REPO --title "[UX] [Medium] EvaluationEngine.run() 阻塞等待回车，无法 CI 集成" `
  --label "enhancement" `
  --body @"
## 问题描述

**文件**: \`core/engine.py:71-72\`

\`\`\`python
if self.visualizer:
    input(\"Press Enter to stop visualization...\")  # ← 阻塞主线程
\`\`\`

可视化启动后强制等待用户按回车，无法在 CI/CD 或非交互环境下运行。

## 修复建议

- 添加配置项 \`auto_exit: true\` 跳过等待
- 支持 \`SIGINT\` / \`SIGTERM\` 信号优雅退出
- 生产模式（\`--no-visualize\` 或环境变数 \`CI=true\`）自动禁用阻塞
"@

# Issue #8
& $GH issue create --repo $REPO --title "[Test] [Medium] 测试目录为空，核心模块无单元测试" `
  --label "test" `
  --body @"
## 问题描述

\`tests/\` 目录下只有 \`__init__.py\`，无任何测试文件。核心组件（\`DataFilter\`、\`StandardTaskRunner\`、\`Leaderboard\`、\`Registry\`）均无测试覆盖。

## 修复建议

引入 \`pytest\`，优先为以下模块补充测试：
1. \`core/data_filter.py\` — DataFilter.apply() 三种过滤模式
2. \`core/registry.py\` — 注册/获取/列表
3. \`evaluators/base.py\` — AccuracyEvaluator
4. \`core/leaderboard.py\` — pretty_print 不崩溃

在 \`pyproject.toml\` 或 \`setup.cfg\` 中配置 pytest，并在 CI 中运行 \`pytest tests/\`。
"@

# Issue #9
& $GH issue create --repo $REPO --title "[Bug] [Medium] core/auto_import.py 导入失败无日志，调试困难" `
  --label "bug" `
  --body @"
## 问题描述

**文件**: \`core/auto_import.py\`

\`\`\`python
def auto_import(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + \".\"):
        importlib.import_module(module_name)  # ← 任何 ImportError 导致整个流程中断，无日志
\`\`\`

某个 adapter 模块导入失败会直接抛异常，且没有任何日志提示是哪个文件出了问题。

## 修复建议

\`\`\`python
def auto_import(package_name, verbose=False):
    package = importlib.import_module(package_name)
    for _, module_name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + \".\"):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f\"Failed to import {module_name}: {e}\")
            if verbose:
                raise
\`\`\`
"@

# Issue #10
& $GH issue create --repo $REPO --title "[Doc] [Low] 配置字段名不统一：path vs data_path" `
  --label "documentation" `
  --body @"
## 问题描述

| 位置 | 字段名 |
|------|--------|
| \`configs/test.yaml:8\` | \`data_path: E:/eval/...\` |
| \`README.md:83\` | \`path: \"data/chinese_simpleqa.json\"\` |
| \`adapter/chinese_simpleqa/chinese_simpleqa.py:26\` | 读取 \`config.get('data_path')\` |

README 示例文档使用 \`path\`，但代码实际读取 \`data_path\`，用户按文档配置会报 \`ValueError: requires 'data_path' config\`。

## 修复建议

统一字段名为 \`data_path\`，并更新 README 示例代码。
"@

Write-Host ""
Write-Host "=== 全部完成 ===" -ForegroundColor Green
