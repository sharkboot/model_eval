r"""
ChineseSimpleQA 适配器
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("ChineseSimpleQA", "dataset")
class ChineseSimpleQADataset(BaseDataset):
    """ChineseSimpleQA 数据集"""

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("ChineseSimpleQA requires 'data_path' config")
        self.dataset_name = "ChineseSimpleQA"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        return DataItem(
            id=self.build_id(data_item.get('question_id', data_item.get('id', ''))),
            prompt=data_item.get('question', ''),
            reference=data_item.get('answer', ''),
            metadata={},
            category=[
                data_item.get('primary_category'),
                data_item.get('secondary_category')
            ]
        )
