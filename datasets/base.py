class BaseDataset:
    def __init__(self, config):
        self.config = config
        self.data = []
    
    def load(self):
        raise NotImplementedError("Subclass must implement load method")
    
    def get_data(self):
        return self.data
    
    def get_dataset_info(self):
        raise NotImplementedError("Subclass must implement get_dataset_info method")
