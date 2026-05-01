"""
C-Eval 适配器

数据集来源: https://huggingface.co/datasets/ceval/ceval-exam
论文: C-Eval: A Multi-Level Multi-Domain Chinese Evaluation Benchmark

C-Eval 是一个中文大语言模型评估基准，包含13948道选择题，
涵盖4个难度级别和52个学科。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("CEval", "dataset")
class CEvalDataset(BaseDataset):
    """
    C-Eval 数据集适配器

    数据格式:
    - id: 问题ID
    - question: 问题描述
    - A/B/C/D: 选项
    - answer: 答案
    - explanation: 解析
    - category: 学科类别
    - difficulty: 难度 (easy/medium/hard/vhard)
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("CEval requires 'data_path' config")
        self.dataset_name = "CEval"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        return read_file(self.data_path)

    def preprocess(self, data_item):
        # 构建完整问题（包含选项）
        question = data_item.get('question', '')
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            if opt in data_item and data_item[opt]:
                options.append(f"{opt}. {data_item[opt]}")
        if options:
            question = f"{question}\n" + "\n".join(options)

        answer = data_item.get('answer', '')
        category = data_item.get('category', [])
        if isinstance(category, str):
            category = [category]
        difficulty = data_item.get('difficulty', 'medium')

        return DataItem(
            id=self.build_id(data_item.get('id', '')),
            prompt=question,
            reference=answer,
            metadata={
                'explanation': data_item.get('explanation', ''),
                'difficulty': difficulty,
            },
            category=category,
            difficulty=difficulty
        )


@Registry.register("CEvalHard", "dataset")
class CEvalHardDataset(CEvalDataset):
    """
    C-Eval 高难度子集

    只包含 difficulty=hard 或 difficulty=vhard 的问题
    """

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "CEvalHard"

    def preprocess(self, data_item):
        difficulty = data_item.get('difficulty', 'medium')
        if difficulty not in ['hard', 'vhard']:
            return None
        return super().preprocess(data_item)