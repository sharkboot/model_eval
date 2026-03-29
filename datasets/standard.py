from .base import BaseDataset

class StandardDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = config.get('dataset_name')
        self.split = config.get('split', 'test')
    
    def load(self):
        # 模拟数据集加载，不依赖modelscope
        print(f"Loading standard dataset: {self.dataset_name} ({self.split})")
        # 生成模拟数据
        self.data = [
            {'prompt': f'Question {i} from {self.dataset_name}', 'answer': f'Answer {i}'}
            for i in range(5)
        ]
    
    def get_dataset_info(self):
        return {
            'dataset_type': 'standard',
            'dataset_name': self.dataset_name,
            'split': self.split,
            'data_size': len(self.data)
        }

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
