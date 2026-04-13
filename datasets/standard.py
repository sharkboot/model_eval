from .base import BaseDataset
from core.base import DataItem

class StandardDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = config.get('dataset_name')
        self.split = config.get('split', 'test')
    
    def load(self):
        # 模拟数据集加载，不依赖modelscope
        print(f"Loading standard dataset: {self.dataset_name} ({self.split})")
        # 生成模拟数据
        raw_data = [
            {'prompt': f'Question {i} from {self.dataset_name}', 'answer': f'Answer {i}'}
            for i in range(5)
        ]
        return [self.preprocess(item) for item in raw_data]
    
    def preprocess(self, data_item):
        prompt = data_item.get('prompt', data_item.get('question', ''))
        reference = data_item.get('answer', '')
        metadata = {
            key: value for key, value in data_item.items()
            if key not in ['prompt', 'question', 'answer']
        }
        metadata['dataset_name'] = self.dataset_name
        metadata['split'] = self.split
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=prompt,
            reference=reference,
            metadata=metadata,
            category=[self.dataset_name]
        )

class MMLUDataset(StandardDataset):
    def __init__(self, config):
        config['dataset_name'] = config.get('dataset_name', 'mmlu')
        super().__init__(config)

class GSM8KDataset(StandardDataset):
    def __init__(self, config):
        config['dataset_name'] = config.get('dataset_name', 'gsm8k')
        super().__init__(config)

class CEvalDataset(StandardDataset):
    def __init__(self, config):
        config['dataset_name'] = config.get('dataset_name', 'c-eval')
        super().__init__(config)

class HumanEvalDataset(StandardDataset):
    def __init__(self, config):
        config['dataset_name'] = config.get('dataset_name', 'humaneval')
        super().__init__(config)
