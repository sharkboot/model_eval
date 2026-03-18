class BaseBackend:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, model, dataset):
        raise NotImplementedError("Subclass must implement evaluate method")
    
    def get_backend_info(self):
        raise NotImplementedError("Subclass must implement get_backend_info method")
