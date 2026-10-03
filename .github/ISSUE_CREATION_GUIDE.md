# GitHub Issue 创建指南

## 问题原因

从 git credential manager 获取的 token **只有读权限**（scopes: `[]`），无法创建 issue/label/milestone。

## 解决方案（二选一）

---

### 方案 A：使用 gh CLI（推荐）

1. 打开 PowerShell，确保已登录：
   ```powershell
   & "C:\Program Files\GitHub CLI\gh.exe" auth status
   ```

2. 如果没有登录，运行：
   ```powershell
   & "C:\Program Files\GitHub CLI\gh.exe" auth login
   # 选择 GitHub.com → HTTPS → 登录浏览器授权
   ```

3. 运行创建脚本：
   ```powershell
   cd E:\LLM\model_eval
   & "C:\Program Files\GitHub CLI\gh.exe" issue create --repo sharkboot/model_eval --title "测试" --body "test"
   ```

---

### 方案 B：创建 Personal Access Token (PAT)

1. 打开 https://github.com/settings/tokens/new
2. 填写：
   - Note: `model_eval_write`
   - Expiration: 90 days
   - Scopes: 勾选 `repo`（完整仓库权限）
3. 点击 Generate token
4. 复制 token（以 `ghp_` 或 `github_pat_` 开头）
5. 运行：
   ```powershell
   $env:GH_TOKEN = "你的token"
   python .github/create_issues.py
   ```

---

## 快速一键命令（复制粘贴到 PowerShell）

```powershell
# 先登录（如果还没登录）
& "C:\Program Files\GitHub CLI\gh.exe" auth status

# 创建 labels
& "C:\Program Files\GitHub CLI\gh.exe" label create critical --color B60205 --description "Critical bug or security issue" --repo sharkboot/model_eval
& "C:\Program Files\GitHub CLI\gh.exe" label create security --color DA321F --description "Security vulnerability" --repo sharkboot/model_eval
& "C:\Program Files\GitHub CLI\gh.exe" label create test --color 0E8A16 --description "Test coverage" --repo sharkboot/model_eval
& "C:\Program Files\GitHub CLI\gh.exe" label create refactor --color C5DEF5 --description "Code refactoring" --repo sharkboot/model_eval

# 创建 milestone
& "C:\Program Files\GitHub CLI\gh.exe" api "repos/sharkboot/model_eval/milestones" -f title="v0.2.0 稳定性与质量改进" -f description="修复 critical bugs、增加测试覆盖" -f state="open" -f due_on="2026-12-31" --method POST

# 创建 issue（示例第一个）
& "C:\Program Files\GitHub CLI\gh.exe" issue create --repo sharkboot/model_eval --title "[Bug] [Critical] core/base.py ModelOutput 类体内混入装饰器死代码" --label "bug,critical" --body "## 问题描述\n\n**文件**: \`core/base.py:10-12\`\n\n```python\nclass ModelOutput:\n    def get_text(self) -> str:\n        cls._registry.setdefault(group, {})[name] = obj  # 死代码\n        return obj\n    return wrapper\n```\n\n这三行代码从 \`@Registry.register\` 装饰器误粘贴进 \`ModelOutput\`，导致 SyntaxError。\n\n## 修复建议\n删除死代码，仅保留 \`get_text()\` 和 \`get_messages()\` 方法。"
```

---

## 10 个 Issue 完整列表

| # | 优先级 | 标题 | Labels |
|---|--------|------|--------|
| 1 | 🔴 Critical | core/base.py ModelOutput 类体内混入装饰器死代码 | bug, critical |
| 2 | 🔴 Critical | models/test_minmax.py 硬编码 API Key 已泄露 | bug, critical, security |
| 3 | 🔴 Critical | core/data_filter.py apply() 方法语法错误，过滤逻辑不生效 | bug, critical |
| 4 | 🟠 High | evaluators/base.py 评估器签名不一致且返回类型错误 | bug |
| 5 | 🟠 High | StandardTaskRunner 并发写入无缓冲，崩溃时数据可能丢失 | enhancement, refactor |
| 6 | 🟠 High | core/leaderboard.py 非数值 metric 导致格式化 TypeError | bug |
| 7 | 🟡 Medium | EvaluationEngine.run() 阻塞等待回车，无法 CI 集成 | enhancement |
| 8 | 🟡 Medium | 测试目录为空，核心模块无单元测试 | test |
| 9 | 🟡 Medium | core/auto_import.py 导入失败无日志，调试困难 | bug |
| 10 | 🟢 Low | 配置字段名不统一：path vs data_path | documentation |
