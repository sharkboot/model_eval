class BaseBackend:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, model, dataset):
        raise NotImplementedError("Subclass must implement evaluate method")
    
    def get_backend_info(self):
        raise NotImplementedError("Subclass must implement get_backend_info method")
    
    def execute(self, model, case):
        """执行案例评估
        输入：案例dict，包含prompt、answer、metadata字段
        输出：评估分数
        """
        raise NotImplementedError("Subclass must implement execute method")
