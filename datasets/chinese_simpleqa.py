from .base import BaseDataset
import os
from utils.data_reader import read_file

class ChineseSimpleQADataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        self.data_type = 'chinese_simpleqa'
    
    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = read_file(self.data_path)
    
    def get_dataset_info(self):
        return {
            'dataset_type': 'custom',
            'data_type': self.data_type,
            'data_path': self.data_path,
            'data_size': len(self.data)
        }
    
    def convert_to_case(self, item):
        """将数据项转换为标准案例格式"""
        prompt = item.get('question', '')
        answer = item.get('answer', '')
        metadata = {
            'id': item.get('id', ''),
            'primary_category': item.get('primary_category', ''),
            'secondary_category': item.get('secondary_category', ''),
            'urls': item.get('urls', [])
        }
        return {
            'prompt': prompt,
            'answer': answer,
            'metadata': metadata
        }
