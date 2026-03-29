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
    
    def convert_to_case(self, item):
        """将数据项转换为标准案例格式
        每个案例是一个dict，包含以下字段：
        - prompt: 输入提示
        - answer: 参考答案
        - metadata: 其他字段
        """
        raise NotImplementedError("Subclass must implement convert_to_case method")
