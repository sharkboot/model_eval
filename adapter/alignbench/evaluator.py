"""
AlignBench 评估器

官方 AlignBench (THUDM/AlignBench, ACL 2024) 评分方案：
- 5 大类：准确性、指令遵循、逻辑性、实用性、完整性
- 每类 20 题共 100 题
- LLM-as-Judge 多维度评分（1-5 分）后聚合
- 官方没有单一 "accuracy" 指标

TODO: 此评估器为简化实现，实际评分应调用 LLM 按官方 rubric 逐维度打分。
      参考官方评测代码: https://github.com/THUDM/AlignBench
"""

import logging
from typing import Dict, List

from core.base import DataItem
from core.registry import Registry
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)
logger.warning(
    "[alignbench_judge] 此评估器为简化实现，使用启发式评分。"
    "请参考官方 AlignBench 评测代码实现真实 LLM-as-Judge 评分。"
)


@Registry.register("alignbench_judge", "evaluator")
class AlignBenchJudgeEvaluator(BaseEvaluator):
    """AlignBench LLM-as-Judge 评估器（简化实现）"""

    def __init__(self, config):
        super().__init__(config)
        self.judge_model = config.get("judge_model", "claude")

    def evaluate(self, pred: str, item: DataItem) -> dict:
        """评估模型输出，返回多维度分数。

        Returns:
            {
                "accuracy": float,     # 综合得分 (0-1)
                "dimension_scores": dict,  # 各维度原始分 (1-5)
            }
        """
        reference = str(item.reference)
        scores = self._judge_output(pred, reference)

        return {
            "accuracy": scores.get("overall", 0.0),
            "dimension_scores": scores,
        }

    def _judge_output(self, pred: str, reference: str) -> Dict[str, float]:
        """评判输出质量（简化启发式版本）。

        TODO: 替换为真实 LLM-as-Judge 调用，按官方 rubric 评分。
        """
        if not pred or len(pred.strip()) < 10:
            base_score = 0.1
        else:
            base_score = min(1.0, 0.5 + 0.1 * (len(pred) / max(len(reference), 1)))

        # 各维度使用相同基础分（真实实现应分别打分）
        return {
            "overall": base_score,
            "factuality": base_score,
            "helpfulness": base_score,
            "clarity": base_score,
            "logic": base_score,
        }


@Registry.register("alignbench_fact", "evaluator")
class AlignBenchFactEvaluator(BaseEvaluator):
    """AlignBench 事实性评估器（简化实现）"""

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
        """计算词重叠度 (Jaccard similarity)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0
