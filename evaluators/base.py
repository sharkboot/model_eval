from abc import ABC, abstractmethod
from core.base import DataItem, EvaluationResult
from core.registry import Registry


class BaseEvaluator(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def evaluate(self, pred: str, data_item: DataItem) -> dict:
        """对比参考答案与模型输出，返回评估结果（dict 格式 metrics）"""
        pass


@Registry.register("accuracy", "evaluator")
class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, pred: str, data_item: DataItem) -> dict:
        acc = 1.0 if pred.strip() == str(data_item.reference).strip() else 0.0
        return {"accuracy": acc}