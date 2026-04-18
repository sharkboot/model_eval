import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("ChineseSimpleQA", "dataset")
class ChineseSimpleQADataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path', config.get('path'))

    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        raw_data = read_file(self.data_path)
        return [self.preprocess(item) for item in raw_data]

    def preprocess(self, data_item):
        prompt = data_item.get('question', '')
        reference = data_item.get('answer', '')
        metadata = {
        }
        return DataItem(
            id=str(hash(str(data_item.get("id", "")))),
            prompt=prompt,
            reference=reference,
            metadata=metadata,
            category=[data_item.get("primary_category"), data_item.get("secondary_category")]
        )
