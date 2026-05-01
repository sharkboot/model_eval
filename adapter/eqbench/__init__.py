"""
EqBench 适配器

数据集来源: https://github.com/EqBench/EqBench
论文: EqBench: A Mathematical Reasoning Benchmark with Generative Language Models

EqBench 评估大语言模型在数学推理方面的能力，包含多种难度级别的问题。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("EqBench", "dataset")
class EqBenchDataset(BaseDataset):
    """
    EqBench 数据集适配器

    数据格式:
    - question: 问题描述
    - answer: 答案
    - solution: 解答过程
    - difficulty: 难度 (1-10 级别，1-3简单，4-6中等，7-10困难)
    - type: 问题类型
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("EqBench requires 'data_path' config")
        self.dataset_name = "EqBench"
        self.min_difficulty = config.get('min_difficulty', 0)
        self.max_difficulty = config.get('max_difficulty', 10)

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = (
            data_item.get('question') or
            data_item.get('problem') or
            data_item.get('prompt', '')
        )

        answer = (
            data_item.get('answer') or
            data_item.get('solution', '')
        )

        difficulty = data_item.get('difficulty', 5)
        if isinstance(difficulty, (int, float)):
            if difficulty <= 3:
                difficulty_str = 'easy'
            elif difficulty <= 6:
                difficulty_str = 'medium'
            else:
                difficulty_str = 'hard'
        else:
            difficulty_str = difficulty

        if isinstance(difficulty, (int, float)):
            if difficulty < self.min_difficulty or difficulty > self.max_difficulty:
                return None

        return DataItem(
            id=self.build_id(data_item.get('id', data_item.get('question_id', ''))),
            prompt=question,
            reference=answer,
            metadata={
                'difficulty': difficulty,
                'type': data_item.get('type', ''),
                'solution': data_item.get('solution', ''),
            },
            category=['math', 'reasoning', data_item.get('type', '')],
            difficulty=difficulty_str
        )


@Registry.register("EqBenchEasy", "dataset")
class EqBenchEasyDataset(EqBenchDataset):
    """EqBench 简单题目子集 (difficulty 1-3)"""

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "EqBenchEasy"
        self.min_difficulty = 1
        self.max_difficulty = 3


@Registry.register("EqBenchMedium", "dataset")
class EqBenchMediumDataset(EqBenchDataset):
    """EqBench 中等题目子集 (difficulty 4-6)"""

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "EqBenchMedium"
        self.min_difficulty = 4
        self.max_difficulty = 6


@Registry.register("EqBenchHard", "dataset")
class EqBenchHardDataset(EqBenchDataset):
    """EqBench 困难题目子集 (difficulty 7-10)"""

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "EqBenchHard"
        self.min_difficulty = 7
        self.max_difficulty = 10