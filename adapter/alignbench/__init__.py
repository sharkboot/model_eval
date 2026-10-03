"""
AlignBench 适配器

数据集来源: https://github.com/THUDM/AlignBench
论文: AlignBench: Evaluating Aligned Large Language Models in Chinese

AlignBench 是中文对齐能力基准，覆盖 5 大类（准确性、指令遵循、逻辑性、实用性、完整性），
每类 20 题共 100 题。官方评分采用 LLM-as-Judge 多维度评分。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("AlignBench", "dataset")
class AlignBenchDataset(BaseDataset):
    """AlignBench 数据集适配器"""

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
        # 官方 schema: id / category / subcategory / question / reference
        # 注意：没有 question_id 字段，使用 id 作为唯一标识
        return DataItem(
            id=self.build_id(data_item.get('id', '')),
            prompt=data_item.get('question', ''),
            reference=data_item.get('reference', ''),
            metadata={
                'category': data_item.get('category', ''),
                'subcategory': data_item.get('subcategory', ''),
            },
            category=[
                data_item.get('category', ''),
                data_item.get('subcategory', ''),
            ],
        )
