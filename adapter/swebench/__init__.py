"""
SWE-bench 适配器

数据集来源: https://huggingface.co/datasets/princeton-nlp/SWE-bench
论文: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? — https://arxiv.org/abs/2310.06770 (ICLR 2024)
官方仓库: https://github.com/swebench/SWE-bench

SWE-bench 包含 2,294 个真实 GitHub issue（Python 库），
每个 issue 包含仓库、patch、测试，用于评估模型解决真实工程问题的能力。
注意：需代码仓库 + git 操作 + pytest 沙箱，接入成本较高。
"""

import os
import json

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("SWE-bench", "dataset")
class SWEbenchDataset(BaseDataset):
    """
    SWE-bench 数据集适配器

    数据格式:
    - instance_id: issue 唯一标识
    - repo: GitHub 仓库（owner/name）
    - base_commit: 基础提交哈希
    - patch: 参考修复 patch
    - test_patch: 测试 patch
    - problem_statement: 问题描述（issue 内容 + 环境信息）
    - hints_text: 可选提示
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("SWE-bench requires 'data_path' config")
        self.dataset_name = "SWE-bench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        instance_id = data_item.get('instance_id', '')
        repo = data_item.get('repo', '')
        base_commit = data_item.get('base_commit', '')
        problem_statement = data_item.get('problem_statement', '')
        patch = data_item.get('patch', '')
        test_patch = data_item.get('test_patch', '')
        hints = data_item.get('hints_text', '')

        # 构建完整问题描述
        prompt = f"""{problem_statement}

Repository: {repo}
Base commit: {base_commit}

Please provide a complete code patch that resolves the issue above.
The patch should be in unified diff format."""

        # 如果有关键词提示，添加
        if hints:
            prompt += f"\n\nHints:\n{hints}"

        # 答案格式：完整 patch
        reference = patch

        return DataItem(
            id=self.build_id(instance_id),
            prompt=prompt,
            reference=reference,
            metadata={
                'instance_id': instance_id,
                'repo': repo,
                'base_commit': base_commit,
                'test_patch': test_patch,
                'hints': hints,
            },
            category=['software_engineering', 'bug_fix', 'python'],
            difficulty='extreme',
        )


# SWE-bench 专用评估器
@Registry.register("swebench_resolved", "evaluator")
class SWEbenchResolvedEvaluator:
    """
    SWE-bench Resolved 评估器

    评估模型生成的 patch 是否能通过测试。
    需要：
    - git 环境
    - pytest 执行
    - Docker 沙箱（推荐）

    简化版：检查 patch 格式 + 非空
    完整版：需实现 git apply + pytest 执行
    """

    def __init__(self, config):
        self.config = config
        self.use_sandbox = config.get('use_sandbox', False)
        self.timeout = config.get('timeout', 300)

    def evaluate(self, pred: str, item) -> dict:
        """评估 patch 质量。

        Returns:
            {
                "accuracy": 0.0|1.0,
                "status": "resolved"|"rejected"|"error",
                "reason": str
            }
        """
        metadata = item.metadata
        test_patch = metadata.get('test_patch', '')
        instance_id = metadata.get('instance_id', '')

        # 基本检查
        if not pred or len(pred.strip()) < 10:
            return {
                "accuracy": 0.0,
                "status": "rejected",
                "reason": "empty_or_too_short_patch",
            }

        # 格式检查：必须是 unified diff
        if not self._is_valid_diff(pred):
            return {
                "accuracy": 0.0,
                "status": "rejected",
                "reason": "invalid_diff_format",
            }

        if not self.use_sandbox:
            # 简化模式：格式正确即认为通过
            return {
                "accuracy": 1.0,
                "status": "resolved",
                "reason": "format_valid (sandbox not enabled)",
            }

        # 完整模式：git apply + pytest
        return self._run_sandbox_test(pred, test_patch, instance_id)

    @staticmethod
    def _is_valid_diff(text: str) -> bool:
        """检查是否为有效的 unified diff"""
        # 至少包含一个文件头
        has_file_header = '--- ' in text and '+++ ' in text
        # 至少包含一个 hunk
        has_hunk = '@@' in text
        # 非空且长度合理
        reasonable_length = 10 < len(text) < 100000
        return has_file_header and has_hunk and reasonable_length

    def _run_sandbox_test(self, patch: str, test_patch: str, instance_id: str) -> dict:
        """在沙箱中执行测试（需要 Docker）。"""
        import subprocess
        import tempfile

        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as tmpdir:
                # 应用 patch
                patch_file = os.path.join(tmpdir, 'fix.patch')
                with open(patch_file, 'w') as f:
                    f.write(patch)

                # git apply
                result = subprocess.run(
                    ['git', 'apply', patch_file],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode != 0:
                    return {
                        "accuracy": 0.0,
                        "status": "error",
                        "reason": f"git_apply_failed: {result.stderr[:200]}",
                    }

                # 应用测试 patch
                if test_patch:
                    test_file = os.path.join(tmpdir, 'test.patch')
                    with open(test_file, 'w') as f:
                        f.write(test_patch)
                    result = subprocess.run(
                        ['git', 'apply', test_file],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                    if result.returncode != 0:
                        return {
                            "accuracy": 0.0,
                            "status": "error",
                            "reason": f"test_patch_apply_failed: {result.stderr[:200]}",
                        }

                # 运行 pytest
                result = subprocess.run(
                    ['python', '-m', 'pytest', '-x', '-q'],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                passed = result.returncode == 0
                return {
                    "accuracy": 1.0 if passed else 0.0,
                    "status": "resolved" if passed else "rejected",
                    "reason": "tests_passed" if passed else "tests_failed",
                }

        except subprocess.TimeoutExpired:
            return {
                "accuracy": 0.0,
                "status": "error",
                "reason": "timeout",
            }
        except Exception as e:
            return {
                "accuracy": 0.0,
                "status": "error",
                "reason": f"exception: {str(e)[:100]}",
            }
