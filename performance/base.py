class BasePerformanceTest:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.dataset = None
    
    def setup(self, model, dataset):
        self.model = model
        self.dataset = dataset
    
    def run(self):
        raise NotImplementedError("Subclass must implement run method")
    
    def get_metrics(self):
        raise NotImplementedError("Subclass must implement get_metrics method")
