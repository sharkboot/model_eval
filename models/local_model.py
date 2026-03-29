from .base import BaseModel

class LocalModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.get('model_name')
        self.device = config.get('device', 'cpu')
        self._load_model()
    
    def _load_model(self):
        # 模拟模型加载，不依赖modelscope
        print(f"Loading local model: {self.model_name} on {self.device}")
    
    def generate(self, prompt, **kwargs):
        # 模拟模型生成
        return f"Generated response for: {prompt}"
    
    def get_model_info(self):
        return {
            'model_type': 'local',
            'model_name': self.model_name,
            'device': self.device
        }
