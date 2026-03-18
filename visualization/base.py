class BaseVisualizer:
    def __init__(self, config):
        self.config = config
    
    def setup(self):
        raise NotImplementedError("Subclass must implement setup method")
    
    def visualize(self, data):
        raise NotImplementedError("Subclass must implement visualize method")
    
    def save(self):
        raise NotImplementedError("Subclass must implement save method")
