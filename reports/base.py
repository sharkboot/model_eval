class BaseReport:
    def __init__(self, config):
        self.config = config
        self.results = []
    
    def add_result(self, result):
        self.results.append(result)
    
    def generate(self):
        raise NotImplementedError("Subclass must implement generate method")
    
    def save(self, path):
        raise NotImplementedError("Subclass must implement save method")
