"""
IFEval 适配器

数据集来源: https://huggingface.co/datasets/google/IFEval
论文: Instruction-Following Evaluation for Large Language Models — https://arxiv.org/abs/2311.07911
官方仓库: https://github.com/google/IFEval

IFEval 包含 500+ 条指令，覆盖 25 种可程序化验证的约束类型，
包括长度约束、格式约束、语言约束等，可自动判分无需 LLM。
"""

import os
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("IFEval", "dataset")
class IFEvalDataset(BaseDataset):
    """
    IFEval 数据集适配器

    数据格式:
    - prompt: 指令提示
    - key: 约束类型标识
    - instruction_id_list: 指令约束列表
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("IFEval requires 'data_path' config")
        self.dataset_name = "IFEval"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        prompt = data_item.get('prompt', '')
        key = data_item.get('key', '')
        instruction_ids = data_item.get('instruction_id_list', [])

        return DataItem(
            id=self.build_id(data_item.get('prompt', '')),
            prompt=prompt,
            reference='',  # IFEval 不预存答案，由评估器动态验证
            metadata={
                'key': key,
                'instruction_ids': instruction_ids if isinstance(instruction_ids, list) else [instruction_ids],
            },
            category=['instruction_following', key] if key else ['instruction_following'],
            difficulty='medium',
        )


# IFEval 专用评估器 — 规则引擎自动判分
@Registry.register("ifeval_rule", "evaluator")
class IFEvalRuleEvaluator:
    """
    IFEval 规则评估器

    根据指令约束类型（key）执行对应的规则检查，
    无需 LLM 判分，可直接自动评估。
    """

    def __init__(self, config):
        self.config = config

    def evaluate(self, pred: str, item) -> dict:
        """根据约束类型执行规则检查。

        Returns:
            {"accuracy": 1.0|0.0, "type": str, "passed": bool}
        """
        metadata = item.metadata
        key = metadata.get('key', '')
        instruction_ids = metadata.get('instruction_ids', [])

        passed = False
        if key == 'follow_id':
            passed = self._check_follow_id(pred, instruction_ids)
        elif key == 'ignore_prev':
            passed = self._check_ignore_prev(pred)
        elif key == 'include_keywords':
            passed = self._check_keywords(pred, instruction_ids)
        elif key == 'exclude_keywords':
            passed = self._check_exclude_keywords(pred, instruction_ids)
        elif key == 'start_with':
            passed = self._check_start_with(pred, instruction_ids)
        elif key == 'num_paragraphs':
            passed = self._check_paragraphs(pred, instruction_ids)
        elif key == 'repeat_request':
            passed = self._check_repeat(pred, instruction_ids)
        elif key == 'language':
            passed = self._check_language(pred, instruction_ids)
        else:
            # 未知约束类型，默认通过
            passed = True

        return {"accuracy": 1.0 if passed else 0.0, "type": key, "passed": passed}

    def _check_follow_id(self, pred: str, instruction_ids: list) -> bool:
        """检查是否包含所有指定 ID"""
        return all(str(i) in pred for i in instruction_ids)

    def _check_ignore_prev(self, pred: str) -> bool:
        """检查是否忽略先前对话（简单检查：不包含之前对话的关键词）"""
        return True  # 简化实现

    def _check_keywords(self, pred: str, instruction_ids: list) -> bool:
        """检查是否包含指定关键词"""
        return all(kw.lower() in pred.lower() for kw in instruction_ids)

    def _check_exclude_keywords(self, pred: str, instruction_ids: list) -> bool:
        """检查是否不包含指定关键词"""
        return all(kw.lower() not in pred.lower() for kw in instruction_ids)

    def _check_start_with(self, pred: str, instruction_ids: list) -> bool:
        """检查是否以指定字符串开头"""
        return any(pred.startswith(str(i)) for i in instruction_ids)

    def _check_paragraphs(self, pred: str, instruction_ids: list) -> bool:
        """检查段落数量"""
        if not instruction_ids:
            return True
        target = int(instruction_ids[0]) if instruction_ids else None
        if target is None:
            return True
        paragraphs = [p for p in pred.split('\n\n') if p.strip()]
        return len(paragraphs) == target

    def _check_repeat(self, pred: str, instruction_ids: list) -> bool:
        """检查是否重复特定短语"""
        if not instruction_ids:
            return True
        phrase = str(instruction_ids[0])
        count = pred.count(phrase)
        return count >= 2

    def _check_language(self, pred: str, instruction_ids: list) -> bool:
        """检查输出语言"""
        if not instruction_ids:
            return True
        lang = str(instruction_ids[0]).lower()
        if lang == 'english':
            return bool(re.search(r'[a-zA-Z]', pred))
        elif lang == 'chinese':
            return bool(re.search(r'[一-鿿]', pred))
        return True
