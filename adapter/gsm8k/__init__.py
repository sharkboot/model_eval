"""
GSM8K 适配器

数据集来源: https://huggingface.co/datasets/openai/gsm8k
论文: Training Verifiers to Solve Math Word Problems — https://arxiv.org/abs/2110.14168 (NeurIPS 2021)
官方仓库: https://github.com/openai/grade-school-math

GSM8K (Grade School Math 8K) 包含 8.5k 小学数学应用题，
涵盖多步算术推理，答案为标准整数。
"""

import os
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("GSM8K", "dataset")
class GSM8KDataset(BaseDataset):
    """
    GSM8K 数据集适配器

    数据格式:
    - question: 问题描述
    - answer: 答案（含 step-by-step 推理，#### 后为最终数字）
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("GSM8K requires 'data_path' config")
        self.dataset_name = "GSM8K"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    @staticmethod
    def extract_final_answer(answer: str) -> str:
        """从 answer 字段提取 #### 后的最终答案。"""
        m = re.search(r'####\s*(\S+)', answer)
        return m.group(1) if m else answer.strip()

    def preprocess(self, data_item):
        question = data_item.get('question', '')
        answer_raw = data_item.get('answer', '')
        reference = self.extract_final_answer(answer_raw)

        return DataItem(
            id=self.build_id(data_item.get('question', '')),
            prompt=question,
            reference=str(reference).strip(),
            metadata={'answer_full': answer_raw},
            category=['math', 'reasoning', 'gsm8k'],
            difficulty='easy',
        )
