from .base import BaseModel
import requests
import json

class APIModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model_name = config.get('model_name')
    
    def generate(self, prompt, **kwargs):
        raise NotImplementedError("Subclass must implement generate method")
    
    def get_model_info(self):
        return {
            'model_type': 'api',
            'model_name': self.model_name,
            'base_url': self.base_url
        }

class OpenAIModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
    
    def generate(self, prompt, **kwargs):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'model': self.model_name,
            'prompt': prompt,
            **kwargs
        }
        response = requests.post(f'{self.base_url}/completions', headers=headers, json=data)
        response.raise_for_status()
        return response.json()

class ClaudeModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.anthropic.com/v1')
    
    def generate(self, prompt, **kwargs):
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        data = {
            'model': self.model_name,
            'prompt': prompt,
            **kwargs
        }
        response = requests.post(f'{self.base_url}/completions', headers=headers, json=data)
        response.raise_for_status()
        return response.json()
