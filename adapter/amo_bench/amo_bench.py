"""
AMO-Bench 适配器

数据集来源: https://huggingface.co/datasets/meituan-longcat/AMO-Bench
论文: AMO-Bench: Large Language Models Still Struggle in High-Difficulty Mathematical Reasoning
GitHub: https://github.com/meituan-longcat/AMO-Bench

AMO-Bench 是一个高级数学推理基准，包含50道人类精心设计的问题，
难度达到 Olympiad（国际数学奥林匹克）级别甚至更高。

数据格式 (来自 HuggingFace):
- 问题来自多个数学领域（代数、几何、数论、组合等）
- 每道题包含问题描述和 boxed 格式的答案
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("AMOBench", "dataset")
class AMOBenchDataset(BaseDataset):
    """
    AMO-Bench 数据集适配器

    评估大语言模型在高级数学推理方面的能力，
    难度达到 Olympiad 级别。
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("AMOBench requires 'data_path' config")
        self.dataset_name = "AMOBench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        return read_file(self.data_path)

    def preprocess(self, data_item):
        """
        预处理 AMO-Bench 数据

        AMO-Bench 数据可能包含:
        - question/problem: 问题描述
        - answer: boxed 格式答案
        - category: 类别
        """
        # 尝试多种可能的字段名
        question = (
            data_item.get('question') or
            data_item.get('problem') or
            data_item.get('prompt', '')
        )

        answer = (
            data_item.get('answer') or
            data_item.get('solution', '')
        )

        # 提取类别
        category = data_item.get('category', [])
        if isinstance(category, str):
            category = [category]

        return DataItem(
            id=self.build_id(data_item.get('id', data_item.get('index', ''))),
            prompt=question,
            reference=answer,
            metadata={
                'category': category,
                'difficulty': 'olympiad',  # AMO-Bench 全部是 Olympiad 级别
            },
            category=category
        )


@Registry.register("AMO", "dataset")
class AMODataset(AMOBenchDataset):
    """
    AMO 数据集适配器 (AMO-Bench 的别名)

    为了兼容性保留，如果用户使用 "AMO" 名称
    """
    pass
