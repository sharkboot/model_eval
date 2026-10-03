"""
AGIEval 适配器

数据集来源: https://huggingface.co/datasets/NLP2CTEvals/AGIEval
论文: AGIEval: A Chinese Evaluation Suite for General Foundational Models — https://arxiv.org/abs/2310.11078
官方仓库: https://github.com/NLP2CTEvals/AGIEval

AGIEval 包含 1,731 题，覆盖高考、考研、公务员、MMLU 等中英文 AGI 能力测试。
评估模型的通用基础能力。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("AGIEval", "dataset")
class AGIEvalDataset(BaseDataset):
    """
    AGIEval 数据集适配器

    数据格式:
    - question: 问题描述
    - options: 选项列表
    - answer: 答案（A/B/C/D 字母）
    - subject: 学科
    - dataset: 来源数据集
    - language: 语言（zh/en）
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("AGIEval requires 'data_path' config")
        self.dataset_name = "AGIEval"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = data_item.get('question', '')
        options = data_item.get('options', [])
        answer = data_item.get('answer', '').strip().upper()
        subject = data_item.get('subject', '')
        dataset = data_item.get('dataset', '')
        language = data_item.get('language', 'zh')

        # 构建 prompt
        prompt = question
        if options:
            options_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options[:4])]
            prompt = f"{question}\n" + "\n".join(options_lines)

        # 标准化答案
        if answer not in ['A', 'B', 'C', 'D']:
            # 尝试数字映射
            try:
                idx = int(answer)
                if 0 <= idx <= 3:
                    answer = chr(65 + idx)
            except ValueError:
                pass

        return DataItem(
            id=self.build_id(data_item.get('id', '')),
            prompt=prompt,
            reference=answer,
            metadata={
                'subject': subject,
                'dataset': dataset,
                'language': language,
                'options': options,
            },
            category=['agi', subject, language] if subject else ['agi', language],
            difficulty='medium',
        )
