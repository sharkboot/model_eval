"""
WritingBench 评估器
"""

from typing import Dict, Any, List

from core.base import DataItem, EvaluationResult
from core.registry import Registry
from adapter.evaluators.base import BaseEvaluator


@Registry.register("writingbench_score", "evaluator")
class WritingBenchScoreEvaluator(BaseEvaluator):
    """
    WritingBench 评分评估器

    基于 LLM-as-Judge 评估模型生成的写作内容。
    每个样本有多个评分标准（checklist），每个标准给出 1-10 分的评分。
    """

    def __init__(self, config):
        super().__init__(config)
        self.judge_model = config.get("judge_model", "claude")

    def evaluate(self, pred: str, item: DataItem) -> EvaluationResult:
        """
        评估模型输出
        """
        checklist = item.reference  # WritingBench 的 reference 是 checklist

        if not isinstance(checklist, list):
            return self._simple_evaluate(pred, item)

        scores = []
        criteria_results = []

        for criterion in checklist:
            if isinstance(criterion, dict):
                criterion_name = criterion.get("name", "unnamed")
                criteria_desc = criterion.get("criteria_description", "")
                score = self._score_single_criterion(pred, criteria_desc)
                scores.append(score)
                criteria_results.append({
                    "name": criterion_name,
                    "score": score
                })

        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "accuracy": avg_score / 10.0,
            "avg_score": avg_score,
            "total_score": sum(scores),
            "max_score": len(scores) * 10,
            "criteria_details": criteria_results
        }

    def _score_single_criterion(self, response: str, criteria: str) -> int:
        """简化评分"""
        score = 5
        if len(response) < 50:
            score = max(1, score - 2)
        elif len(response) > 500:
            score = min(10, score + 1)
        keywords = ["分析", "讨论", "研究", "结论", "方法", "结果"]
        keyword_count = sum(1 for kw in keywords if kw in response)
        if keyword_count > 3:
            score = min(10, score + 1)
        return score

    def _simple_evaluate(self, pred: str, item: DataItem) -> Dict[str, Any]:
        """简单评估"""
        reference = str(item.reference)
        similarity = self._calculate_similarity(pred, reference)
        return {"accuracy": similarity}

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算相似度"""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0


@Registry.register("writingbench", "evaluator")
class WritingBenchSimpleEvaluator(BaseEvaluator):
    """WritingBench 简单评估器"""

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, pred: str, item: DataItem) -> EvaluationResult:
        """简单评估"""
        checklist = item.reference

        if isinstance(checklist, list):
            score = self._calculate_completeness(pred, checklist)
        else:
            score = min(1.0, len(pred) / 500)

        return {"accuracy": score}

    def _calculate_completeness(self, response: str, checklist: List) -> float:
        """计算响应完整性"""
        if not checklist:
            return 1.0

        score = 0.5
        if len(response) > 100:
            score += 0.2
        if len(response) > 500:
            score += 0.2
        if "\n\n" in response:
            score += 0.1

        return min(1.0, score)
