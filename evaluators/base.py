from abc import ABC, abstractmethod
from core.base import DataItem, EvaluationResult

class BaseEvaluator(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def evaluate(self, data_item: DataItem, model_output: str) -> EvaluationResult:
        """对比参考答案与模型输出，返回评估结果"""
        pass