"""
GSM-Plus 适配器

数据集来源: https://huggingface.co/datasets/datalab-to/GSM-Plus
论文: GSM-Plus: The Impact of Input Perturbations and Aggregation Strategies on GSM8K — https://arxiv.org/abs/2410.05228

GSM-Plus 包含 9,000+ 题，是 GSM8K 的抗干扰扩展版本，
加入上下文注入和干扰项，测试模型的鲁棒性。
接入逻辑与 GSM8K 相同。
"""

import os
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("GSM-Plus", "dataset")
class GSMPlusDataset(BaseDataset):
    """
    GSM-Plus 数据集适配器

    数据格式:
    - question: 问题描述（可能含干扰信息）
    - answer: 答案（#### 后为最终数字）
    - context: 上下文（可能含干扰）
    - ground_truth: 标准答案
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("GSM-Plus requires 'data_path' config")
        self.dataset_name = "GSM-Plus"

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
        context = data_item.get('context', '')
        ground_truth = data_item.get('ground_truth', '')

        reference = self.extract_final_answer(answer_raw or ground_truth)

        # GSM-Plus 的特殊性：问题可能包含干扰信息
        prompt = question
        if context:
            prompt = f"{question}\n\n额外信息（可能无关）:\n{context}"

        return DataItem(
            id=self.build_id(data_item.get('question', '')),
            prompt=prompt,
            reference=str(reference).strip(),
            metadata={
                'context': context,
                'ground_truth': ground_truth,
                'answer_full': answer_raw,
            },
            category=['math', 'reasoning', 'gsm_plus', 'robustness'],
            difficulty='medium',
        )
