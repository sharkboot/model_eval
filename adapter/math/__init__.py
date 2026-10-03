"""
MATH 适配器

数据集来源: https://huggingface.co/datasets/HuggingFaceH4/MATH
论文: MATH: Measuring Mathematical Problem Solving — https://arxiv.org/abs/2103.03094 (NeurIPS 2021)
官方仓库: https://github.com/hendrycks/math

MATH 包含 12,500 道数学竞赛题，按 Level 1-5 分层，
覆盖 Algebra、Geometry、Number Theory 等 8 个领域。
答案通常以 \boxed{} 格式给出。
"""

import os
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("MATH", "dataset")
class MATHDataset(BaseDataset):
    """
    MATH 数据集适配器

    数据格式:
    - problem: 问题描述
    - solution: 解答过程
    - answer: 最终答案（通常含 \boxed{}）
    - topic: 领域（Algebra/Geometry/Counting/Probability/Number Theory/Precalculus/Prealgebra/Calculus）
    - level: 难度（1-5）
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("MATH requires 'data_path' config")
        self.dataset_name = "MATH"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    @staticmethod
    def extract_boxed(answer: str) -> str:
        """从 answer 中提取 \boxed{} 内容。"""
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'boxed\{([^}]+)\}',
        ]
        for pat in patterns:
            m = re.search(pat, answer)
            if m:
                return m.group(1).strip()
        return answer.strip()

    def preprocess(self, data_item):
        problem = (
            data_item.get('problem')
            or data_item.get('question')
            or data_item.get('prompt', '')
        )
        answer_raw = data_item.get('answer', '')
        reference = self.extract_boxed(answer_raw)
        topic = data_item.get('topic', 'unknown')
        level = data_item.get('level', 'unknown')

        return DataItem(
            id=self.build_id(data_item.get('problem_id', data_item.get('id', ''))),
            prompt=problem,
            reference=str(reference).strip(),
            metadata={
                'solution': data_item.get('solution', ''),
                'topic': topic,
                'level': level,
            },
            category=['math', 'competition', topic, f'level_{level}'],
            difficulty=level,
        )
