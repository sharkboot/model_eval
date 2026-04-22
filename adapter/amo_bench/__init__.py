"""
AMO-Bench 适配器
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("AMOBench", "dataset")
class AMOBenchDataset(BaseDataset):
    """AMO-Bench 数据集"""

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("AMOBench requires 'data_path' config")
        self.dataset_name = "AMOBench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = (
            data_item.get('question') or
            data_item.get('problem') or
            data_item.get('prompt', '')
        )
        answer = (
            data_item.get('answer') or
            data_item.get('solution', '')
        )
        category = data_item.get('category', [])
        if isinstance(category, str):
            category = [category]

        return DataItem(
            id=self.build_id(data_item.get('id', data_item.get('index', ''))),
            prompt=question,
            reference=answer,
            metadata={'category': category, 'difficulty': 'olympiad'},
            category=category
        )


@Registry.register("AMO", "dataset")
class AMODataset(AMOBenchDataset):
    """AMO 数据集别名"""
    pass
