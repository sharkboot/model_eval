from .base import BaseDataset
import json
import os

class CustomDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        self.data_type = config.get('data_type')
    
    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def get_dataset_info(self):
        return {
            'dataset_type': 'custom',
            'data_type': self.data_type,
            'data_path': self.data_path,
            'data_size': len(self.data)
        }

class MCQDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'mcq'
        super().__init__(config)

class QADataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'qa'
        super().__init__(config)

class FunctionCallDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'function_call'
        super().__init__(config)

class VQADataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'vqa'
        super().__init__(config)
