"""
AlignBench 适配器

数据集来源: THUDM/AlignBench
论文: AlignBench: Multidimensional Evaluation for Large Language Models on
      Scientific and Mathematical Reasoning

数据格式:
- question_id: 问题ID
- category: 一级分类
- subcategory: 二级分类
- question: 问题内容
- reference: 参考答案
- evidences: 证据/参考资料
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("AlignBench", "dataset")
class AlignBenchDataset(BaseDataset):
    """
    AlignBench 数据集适配器

    评估大语言模型在科学和数学推理方面的能力，
    涵盖多个维度的评估。
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("AlignBench requires 'data_path' config")
        self.dataset_name = "AlignBench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        return read_file(self.data_path)

    def preprocess(self, data_item):
        """
        预处理 AlignBench 数据
        """
        return DataItem(
            id=self.build_id(data_item.get('question_id', '')),
            prompt=data_item.get('question', ''),
            reference=data_item.get('reference', ''),
            metadata={
                'question_id': data_item.get('question_id'),
                'evidences': data_item.get('evidences', []),
            },
            category=[
                data_item.get('category'),
                data_item.get('subcategory')
            ]
        )
