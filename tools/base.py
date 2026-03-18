class BaseTool:
    def __init__(self, config):
        self.config = config
    
    def setup(self):
        raise NotImplementedError("Subclass must implement setup method")
    
    def run(self, model):
        raise NotImplementedError("Subclass must implement run method")
    
    def get_results(self):
        raise NotImplementedError("Subclass must implement get_results method")
