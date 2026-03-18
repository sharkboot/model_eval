class BaseModel:
    def __init__(self, config):
        self.config = config
    
    def generate(self, prompt, **kwargs):
        raise NotImplementedError("Subclass must implement generate method")
    
    def get_model_info(self):
        raise NotImplementedError("Subclass must implement get_model_info method")
