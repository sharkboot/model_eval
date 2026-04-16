from .base import BaseModel
from core.base import ModelInput
from .registry import ModelRegistry

@ModelRegistry.register('LocalModel')
class LocalModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.get('model_name')
        self.device = config.get('device', 'cpu')
        self._load_model()
    
    def _load_model(self):
        # 模拟模型加载，不依赖modelscope
        print(f"Loading local model: {self.model_name} on {self.device}")
    
    def generate(self, inputs):
        results = []
        for input_item in inputs:
            prompt = input_item.prompt
            # 模拟模型生成，返回包含参考答案的响应
            if input_item.messages:
                # 如果有对话历史，构建完整的对话上下文
                conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in input_item.messages])
                if "请评估以下模型对问题的回答是否正确" in prompt:
                    # 模拟裁判模型的回答，总是返回1.0
                    results.append("1.0")
                else:
                    results.append(f"Local model response to conversation: {conversation}\nUser: {prompt}")
            else:
                if "请评估以下模型对问题的回答是否正确" in prompt:
                    # 模拟裁判模型的回答，总是返回1.0
                    results.append("1.0")
                elif "首都是哪里" in prompt:
                    results.append("中国的首都是北京")
                elif "2 + 2 =" in prompt:
                    results.append("2 + 2 = 4")
                elif "编程语言" in prompt:
                    results.append("Python是一种编程语言")
                elif "地球的形状" in prompt:
                    results.append("地球的形状是椭圆形")
                elif "最大的行星" in prompt:
                    results.append("太阳系中最大的行星是木星")
                else:
                    results.append(f"Generated response for: {prompt}")
        return results
    
    def get_model_info(self):
        return {
            'model_type': 'local',
            'model_name': self.model_name,
            'device': self.device
        }
