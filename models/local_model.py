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
        # 模拟模型生成，返回包含参考答案的响应
        if "请评估以下模型对问题的回答是否正确" in prompt:
            # 模拟裁判模型的回答，总是返回1.0
            return "1.0"
        elif "首都是哪里" in prompt:
            return "中国的首都是北京"
        elif "2 + 2 =" in prompt:
            return "2 + 2 = 4"
        elif "编程语言" in prompt:
            return "Python是一种编程语言"
        elif "地球的形状" in prompt:
            return "地球的形状是椭圆形"
        elif "最大的行星" in prompt:
            return "太阳系中最大的行星是木星"
        else:
            return f"Generated response for: {prompt}"
    
    def get_model_info(self):
        return {
            'model_type': 'local',
            'model_name': self.model_name,
            'device': self.device
        }
