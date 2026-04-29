"""
AlignBench 评估器
"""

from typing import Dict, List

from core.base import DataItem
from core.registry import Registry
from evaluators.base import BaseEvaluator


@Registry.register("alignbench_judge", "evaluator")
class AlignBenchJudgeEvaluator(BaseEvaluator):
    """AlignBench LLM-as-Judge 评估器"""

    def __init__(self, config):
        super().__init__(config)
        self.judge_model = config.get("judge_model", "claude")

    def evaluate(self, pred: str, item: DataItem) -> dict:
        """评估模型输出"""
        reference = item.reference
        evidences = item.metadata.get("evidences", [])
        scores = self._judge_output(pred, reference, evidences)

        return {
            "accuracy": scores.get("overall", 0.0),
            "dimension_scores": scores
        }

    def _judge_output(self, pred: str, reference: str, evidences: List[str]) -> Dict[str, float]:
        """评判输出质量（简化版本）"""
        overall = 0.5
        if not pred or len(pred.strip()) < 10:
            overall = 0.1
        elif len(pred) > len(reference) * 0.5:
            overall = min(1.0, 0.5 + 0.1 * (len(pred) / max(len(reference), 1)))

        return {
            "overall": overall,
            "factuality": overall,
            "helpfulness": overall,
            "clarity": overall,
            "logic": overall
        }


@Registry.register("alignbench_fact", "evaluator")
class AlignBenchFactEvaluator(BaseEvaluator):
    """AlignBench 事实性评估器"""

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, pred: str, item: DataItem) -> dict:
        """事实性评估"""
        reference = str(item.reference).lower().strip()
        prediction = pred.lower().strip()

        score = 0.0
        if reference in prediction or prediction in reference:
            score = 1.0
        elif self._calculate_overlap(reference, prediction) > 0.7:
            score = 1.0
        elif self._calculate_overlap(reference, prediction) > 0.3:
            score = 0.5

        return {"accuracy": score}

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """计算重叠度"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0
