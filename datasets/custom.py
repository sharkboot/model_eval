from core.registry import Registry
from .base import BaseDataset
import os
from core.data_reader import read_file
from core.base import DataItem

@Registry.register("CustomDataset", "dataset")
class CustomDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path', config.get('path'))
        self.data_type = config.get('data_type')
    
    def load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        raw_data = read_file(self.data_path)
        return [self.preprocess(item) for item in raw_data]
    
    def preprocess(self, data_item):
        prompt = data_item.get('prompt', data_item.get('question', ''))
        reference = data_item.get('answer', '')
        metadata = {
            key: value for key, value in data_item.items()
            if key not in ['prompt', 'question', 'answer']
        }
        # 处理CSV格式中的options字段，将|分隔的字符串转换为列表
        if 'options' in metadata and isinstance(metadata['options'], str):
            metadata['options'] = metadata['options'].split('|')
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=prompt,
            reference=reference,
            metadata=metadata,
            category=[self.data_type] if self.data_type else []
        )

@Registry.register("dummy", "dataset")
class DummyDataset(BaseDataset):
    def load(self):
        return [
            DataItem(id="1", prompt="2+2=?", reference="4"),
            DataItem(id="2", prompt="3+5=?", reference="8"),
        ]
