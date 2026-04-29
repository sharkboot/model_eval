from abc import ABC, abstractmethod
from core.base import DataItem, EvaluationResult
from core.registry import Registry


class BaseEvaluator(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def evaluate(self, pred: str, data_item: DataItem, ) -> EvaluationResult:
        """对比参考答案与模型输出，返回评估结果"""
        pass


@Registry.register("accuracy", "evaluator")
class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, pred: str, item: DataItem):
        acc = 1.0 if pred.strip() == str(item.reference).strip() else 0.0
        return {"accuracy": acc}