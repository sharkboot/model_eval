"""
WritingBench 适配器

数据集来源: https://github.com/THUDM/WritingBench
"""

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("WritingBench", "dataset")
class WritingBenchDataset(BaseDataset):
    """WritingBench 数据集"""

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("WritingBench requires 'data_path' config")
        self.dataset_name = "WritingBench"

    def load_raw_data(self):
        import os
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        query = data_item.get('query', '')
        checklist = data_item.get('checklist', [])

        return DataItem(
            id=self.build_id(data_item.get('index', '')),
            prompt=query,
            reference=checklist,
            metadata={
                'index': data_item.get('index'),
                'domain1': data_item.get('domain1'),
                'domain2': data_item.get('domain2'),
                'lang': data_item.get('lang'),
                'checklist': checklist,
            },
            category=[
                data_item.get('domain1'),
                data_item.get('domain2'),
                data_item.get('lang')
            ]
        )
