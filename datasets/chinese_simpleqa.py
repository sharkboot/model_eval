from .base import BaseDataset
import os
from core.data_reader import read_file
from core.base import DataItem

class ChineseSimpleQADataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        self.data_type = 'chinese_simpleqa'
    
    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        raw_data = read_file(self.data_path)
        return [self.preprocess(item) for item in raw_data]
    
    def preprocess(self, data_item):
        prompt = data_item.get('question', '')
        reference = data_item.get('answer', '')
        metadata = {
            'id': data_item.get('id', ''),
            'primary_category': data_item.get('primary_category', ''),
            'secondary_category': data_item.get('secondary_category', ''),
            'urls': data_item.get('urls', [])
        }
        categories = [self.data_type]
        if metadata.get('primary_category'):
            categories.append(metadata['primary_category'])
        if metadata.get('secondary_category'):
            categories.append(metadata['secondary_category'])
        return DataItem(
            id=data_item.get('id', str(hash(str(data_item)))),
            prompt=prompt,
            reference=reference,
            metadata=metadata,
            category=categories
        )
