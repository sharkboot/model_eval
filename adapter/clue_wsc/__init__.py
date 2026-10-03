"""
CLUEWSC 适配器

中文来源: https://www.cluebenchmarks.com/en/
数据集（HF）: https://huggingface.co/datasets/clue/clue（子集 clue_wsc）
论文: CLUE: A Chinese Language Understanding Evaluation Benchmark — https://aclanthology.org/2020.coling-main.419/（COLING 2020）

CLUEWSC 是 Winograd Schema Challenge 的中文版本，
测试模型的常识推理能力，共 5,089 题。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("CLUEWSC", "dataset")
class CLUEWSCDataset(BaseDataset):
    """
    CLUEWSC 数据集适配器

    数据格式:
    - id: 题目 ID
    - text: 句子（含 A/B 标记）
    - optionA: 选项 A 内容
    - optionB: 选项 B 内容
    - answer: 正确答案（A 或 B）
    - label: 标签
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("CLUEWSC requires 'data_path' config")
        self.dataset_name = "CLUEWSC"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        text = data_item.get('text', '')
        option_a = data_item.get('optionA', '')
        option_b = data_item.get('optionB', '')
        answer = str(data_item.get('answer', '')).upper()
        tid = data_item.get('id', '')

        # 构建 prompt：展示句子和两个选项
        prompt = f"""请判断以下句子中哪个选项是正确的。

{text}

选项：
A. {option_a}
B. {option_b}

请选择 A 或 B。"""

        # 标准化答案
        if answer not in ['A', 'B']:
            if answer == '0':
                answer = 'A'
            elif answer == '1':
                answer = 'B'

        return DataItem(
            id=self.build_id(tid),
            prompt=prompt,
            reference=answer,
            metadata={
                'text': text,
                'option_a': option_a,
                'option_b': option_b,
            },
            category=['chinese', 'reasoning', 'winograd'],
            difficulty='easy',
        )
