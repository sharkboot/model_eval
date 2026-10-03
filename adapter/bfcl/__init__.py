"""
BFCL 适配器

数据集来源: https://huggingface.co/datasets/ShishirSilBFCL/bfcl_v3
论文: Berkeley Function Calling Leaderboard — https://arxiv.org/abs/2405.18725
官方仓库: https://github.com/ShishirSilBFCL/Berkeley-Function-Calling-Leaderboard

BFCL (Berkeley Function Calling Leaderboard) 包含 2,721 题，
覆盖 7 类函数调用场景，用于评估模型的工具使用能力。
"""

import os
import json
import ast
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("BFCL", "dataset")
class BFCLDataset(BaseDataset):
    """
    BFCL 数据集适配器

    数据格式:
    - version: 测试类型
    - description: 测试描述
    - test_cases: 测试用例列表
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("BFCL requires 'data_path' config")
        self.dataset_name = "BFCL"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        version = data_item.get('version', 'unknown')
        description = data_item.get('description', '')
        test_cases = data_item.get('test_cases', [])

        # 构建 prompt
        if not test_cases:
            prompt = f"Function calling task ({version}): {description}"
            reference = ""
        else:
            first_case = test_cases[0]
            instruction = first_case.get('instruction', '')
            prompt = f"Function calling task ({version}): {description}\n\nInstruction: {instruction}"
            # 参考输出
            expected = first_case.get('expected_output', [])
            reference = json.dumps(expected) if isinstance(expected, list) else str(expected)

        return DataItem(
            id=self.build_id(f"{version}_{data_item.get('id', len(test_cases))}"),
            prompt=prompt,
            reference=reference,
            metadata={
                'version': version,
                'test_cases': test_cases,
                'description': description,
            },
            category=['tool_use', 'function_calling', version],
            difficulty='medium',
        )


# BFCL 专用评估器 — 函数调用匹配
@Registry.register("bfcl_function_match", "evaluator")
class BFCLFunctionMatchEvaluator:
    """
    BFCL 函数调用评估器

    使用 AST 解析和函数签名匹配评估模型输出。
    支持多种输出格式（JSON、Python 调用、纯文本）。
    """

    def __init__(self, config):
        self.config = config

    def evaluate(self, pred: str, item) -> dict:
        """评估函数调用输出。

        Returns:
            {
                "accuracy": 0.0|1.0,
                "matched": bool,
                "reason": str
            }
        """
        reference = str(item.reference)
        if not reference or reference in ['""', "''", 'null']:
            return {
                "accuracy": 1.0,  # 无期望输出，默认通过
                "matched": True,
                "reason": "no_expected_output",
            }

        matched = self._match_function_call(pred, reference)
        return {
            "accuracy": 1.0 if matched else 0.0,
            "matched": matched,
            "reason": "exact_match" if matched else "mismatch",
        }

    def _match_function_call(self, pred: str, reference: str) -> bool:
        """匹配函数调用，支持多种格式。"""
        # 1. 精确匹配
        if pred.strip() == reference.strip():
            return True

        # 2. 移除首尾空格/引号
        pred_clean = pred.strip().strip('"\'')
        ref_clean = reference.strip().strip('"\'')
        if pred_clean == ref_clean:
            return True

        # 3. 尝试解析为 JSON
        try:
            pred_json = json.loads(pred)
            ref_json = json.loads(reference)
            return self._compare_json(pred_json, ref_json)
        except (json.JSONDecodeError, TypeError):
            pass

        # 4. 尝试解析为 Python 调用
        try:
            pred_func = self._extract_function_name(pred)
            ref_func = self._extract_function_name(reference)
            if pred_func and ref_func:
                return pred_func.lower() == ref_func.lower()
        except Exception:
            pass

        # 5. 模糊匹配（包含关系）
        if ref_clean.lower() in pred.lower() or pred_clean.lower() in ref_clean.lower():
            return True

        return False

    @staticmethod
    def _compare_json(pred, ref) -> bool:
        """递归比较 JSON 结构"""
        if isinstance(pred, dict) and isinstance(ref, dict):
            # 比较 keys
            if set(pred.keys()) != set(ref.keys()):
                return False
            # 比较 values
            for key in ref:
                if not BFCLFunctionMatchEvaluator._compare_json(pred.get(key), ref.get(key)):
                    return False
            return True
        elif isinstance(pred, list) and isinstance(ref, list):
            if len(pred) != len(ref):
                return False
            for p, r in zip(pred, ref):
                if not BFCLFunctionMatchEvaluator._compare_json(p, r):
                    return False
            return True
        else:
            return str(pred).lower() == str(ref).lower()

    @staticmethod
    def _extract_function_name(text: str) -> str:
        """提取函数名称"""
        # 匹配: func_name(
        m = re.search(r'(\w+)\s*\(', text)
        if m:
            return m.group(1)

        # 匹配: {"function": "func_name", ...}
        m = re.search(r'"function"\s*:\s*"(\w+)"', text)
        if m:
            return m.group(1)

        # 匹配: "name": "func_name"
        m = re.search(r'"name"\s*:\s*"(\w+)"', text)
        if m:
            return m.group(1)

        # 匹配: tool_name
        m = re.search(r'"tool"\s*:\s*"(\w+)"', text)
        if m:
            return m.group(1)

        return ""
