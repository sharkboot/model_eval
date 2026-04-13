from abc import ABC, abstractmethod
from typing import List
from core.base import ModelInput

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def generate(self, inputs: List[ModelInput]) -> List[str]:
        """接收提示词列表，返回生成文本列表"""
        pass
