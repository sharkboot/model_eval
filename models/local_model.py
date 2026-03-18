from .base import BaseModel
from modelscope import AutoModel, AutoTokenizer

class LocalModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.get('model_name')
        self.device = config.get('device', 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, device_map=self.device)
    
    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_model_info(self):
        return {
            'model_type': 'local',
            'model_name': self.model_name,
            'device': self.device
        }
