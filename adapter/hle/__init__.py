"""
HLE 适配器

数据集来源: https://huggingface.co/datasets/cais/hle
官方仓库: https://github.com/centerforaisafety/hle
官网: https://agi.safe.ai/

HLE (Humanity's Last Exam) 由 Center for AI Safety 发布，
包含 2,500 题跨数学/人文/自然科学的前沿问题，
难度极高，适合评估当前 SOTA 模型的极限。
注意：需 HuggingFace 登录同意使用条款。
"""

import os
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("HLE", "dataset")
class HLEDataset(BaseDataset):
    """
    HLE 数据集适配器

    数据格式:
    - problem: 问题描述
    - answer: 答案
    - category: 类别（数学/人文/科学等）
    - subcategory: 子类别
    - canary: 安全字符串（用于追踪泄露）
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("HLE requires 'data_path' config")
        self.dataset_name = "HLE"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        problem = data_item.get('problem', '')
        answer = data_item.get('answer', '')
        category = data_item.get('category', '')
        subcategory = data_item.get('subcategory', '')
        canary = data_item.get('canary', '')

        # 移除 canary（安全字符串）
        prompt = problem.replace(canary, '').strip() if canary else problem

        # 提取最终答案
        import re
        m = re.search(r'boxed\{?([^}]+)\}?', str(answer))
        reference = m.group(1).strip() if m else str(answer).strip()

        return DataItem(
            id=self.build_id(data_item.get('problem_id', data_item.get('id', ''))),
            prompt=prompt,
            reference=str(reference),
            metadata={
                'category': category,
                'subcategory': subcategory,
                'canary': canary,
                'original_answer': answer,
            },
            category=['extreme', category, subcategory] if category else ['extreme'],
            difficulty='extreme',
        )
