"""
HumanEval 适配器

数据集来源: https://huggingface.co/datasets/openai/openai_humaneval
论文: Evaluating Large Language Models Trained on Code — https://arxiv.org/abs/2107.03374
官方仓库: https://github.com/openai/human-eval

HumanEval 包含 164 道 Python 函数补全题，
每题自带 7-20 条单元测试，用于评估代码生成能力。
注意：需要代码执行沙箱环境（pytest），超出纯文本框架范围。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("HumanEval", "dataset")
class HumanEvalDataset(BaseDataset):
    """
    HumanEval 数据集适配器

    数据格式:
    - task_id: 题目 ID（如 HumanEval/0）
    - prompt: 函数签名+docstring（需补全的部分）
    - canonical_solution: 参考实现
    - test: 测试代码
    - setup: 前置代码
    - entry_point: 待测试的函数名
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("HumanEval requires 'data_path' config")
        self.dataset_name = "HumanEval"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        task_id = data_item.get('task_id', '')
        prompt = data_item.get('prompt', '')
        canonical = data_item.get('canonical_solution', '')
        test = data_item.get('test', '')
        setup = data_item.get('setup', '')
        entry_point = data_item.get('entry_point', '')

        # 构建完整问题（包含函数签名）
        full_prompt = f"{prompt}\n\nPlease complete the function below:\n{prompt}"

        return DataItem(
            id=self.build_id(task_id),
            prompt=full_prompt,
            reference=canonical,
            metadata={
                'task_id': task_id,
                'test': test,
                'setup': setup,
                'entry_point': entry_point,
            },
            category=['code', 'python', 'function_completion'],
            difficulty='hard',
        )


# HumanEval 专用评估器 — Pass@k 评分
@Registry.register("humaneval_pass", "evaluator")
class HumanEvalPassEvaluator:
    """
    HumanEval Pass@k 评估器

    执行模型生成的代码，运行单元测试，计算 Pass@k。
    注意：需要代码执行沙箱环境。
    """

    def __init__(self, config):
        self.config = config
        self.timeout = config.get('timeout', 10)
        self.sandbox = config.get('sandbox', False)  # 是否启用沙箱

    def evaluate(self, pred: str, item) -> dict:
        """
        评估代码生成结果。

        返回:
            {"accuracy": 0.0|1.0, "passed_tests": int, "total_tests": int}
        """
        import ast
        import tempfile
        import subprocess
        import os

        metadata = item.metadata
        test_code = metadata.get('test', '')
        setup_code = metadata.get('setup', '')
        entry_point = metadata.get('entry_point', '')

        if not test_code or not entry_point:
            return {"accuracy": 0.0, "error": "missing test or entry_point"}

        # 构建完整代码
        full_code = f"{setup_code}\n{pred}\n{test_code}"

        try:
            # 解析 AST 检查语法
            ast.parse(full_code)

            # 写入临时文件执行
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    ['python', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                passed = result.returncode == 0
                return {
                    "accuracy": 1.0 if passed else 0.0,
                    "passed_tests": 1 if passed else 0,
                    "total_tests": 1,
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }
            finally:
                os.unlink(temp_path)

        except SyntaxError as e:
            return {"accuracy": 0.0, "error": f"syntax_error: {str(e)[:100]}"}
        except subprocess.TimeoutExpired:
            return {"accuracy": 0.0, "error": "timeout"}
        except Exception as e:
            return {"accuracy": 0.0, "error": str(e)[:200]}
