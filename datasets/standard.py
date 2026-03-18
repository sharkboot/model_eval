from .base import BaseDataset
from modelscope import MsDataset

class StandardDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = config.get('dataset_name')
        self.split = config.get('split', 'test')
    
    def load(self):
        dataset = MsDataset.load(self.dataset_name, split=self.split)
        self.data = [item for item in dataset]
    
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
