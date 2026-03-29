from .base import BaseDataset
import os
from utils.data_reader import read_file

class CustomDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        self.data_type = config.get('data_type')
    
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
        prompt = item.get('prompt', item.get('question', ''))
        answer = item.get('answer', '')
        metadata = {
            key: value for key, value in item.items()
            if key not in ['prompt', 'question', 'answer']
        }
        # 处理CSV格式中的options字段，将|分隔的字符串转换为列表
        if 'options' in metadata and isinstance(metadata['options'], str):
            metadata['options'] = metadata['options'].split('|')
        return {
            'prompt': prompt,
            'answer': answer,
            'metadata': metadata
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
