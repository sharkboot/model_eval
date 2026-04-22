"""
AlignBench 适配器
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("AlignBench", "dataset")
class AlignBenchDataset(BaseDataset):
    """AlignBench 数据集"""

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
