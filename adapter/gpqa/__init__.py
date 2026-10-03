"""
GPQA 适配器

数据集来源: https://huggingface.co/datasets/Idavidrein/gpqa
论文: GPQA: A Graduate-Level Google-Proof Q&A Benchmark — https://arxiv.org/abs/2311.12022
官方仓库: https://github.com/idavidrein/gpqa

GPQA 包含 448 道研究生级科学问题（生物/物理/化学），
以高难度著称，要求模型具备深度专业知识。
需登录 HuggingFace 同意使用条款后下载。
"""

import os

from core.base import DataItem
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("GPQA", "dataset")
class GPQADataset(BaseDataset):
    """
    GPQA 数据集适配器

    数据格式:
    - Question: 问题描述
    - A/B/C/D: 选项
    - Correct Answer: 正确答案（A/B/C/D 字母）
    - Domain: 学科领域
    - Subdomain: 子领域
    - Cohort: 考生群体
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("GPQA requires 'data_path' config")
        self.dataset_name = "GPQA"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return self._read_csv(self.data_path)

    @staticmethod
    def _read_csv(path: str):
        """读取 CSV 格式数据"""
        import csv
        with open(path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def preprocess(self, data_item):
        question = data_item.get('Question', '')
        # 选项 A-D
        choices = []
        for letter in ['A', 'B', 'C', 'D']:
            val = data_item.get(letter, '')
            if val:
                choices.append(val)

        question_text = question
        if choices:
            options_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)]
            question_text = f"{question_text}\n" + "\n".join(options_lines)

        # 正确答案
        correct = data_item.get('Correct Answer', '').strip().upper()
        if correct not in ['A', 'B', 'C', 'D']:
            # 尝试解析
            import re
            m = re.match(r'^([ABCD])', correct)
            correct = m.group(1) if m else 'A'

        domain = data_item.get('Domain', '')
        subdomain = data_item.get('Subdomain', '')

        return DataItem(
            id=self.build_id(data_item.get('Question', '')),
            prompt=question_text,
            reference=correct,
            metadata={
                'domain': domain,
                'subdomain': subdomain,
                'cohort': data_item.get('Cohort', ''),
            },
            category=['science', domain, subdomain] if subdomain else ['science', domain],
            difficulty='hard',
        )
