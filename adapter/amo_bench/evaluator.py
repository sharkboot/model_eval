"""
AMO-Bench 评估器
"""

import re

from core.base import DataItem
from core.registry import Registry
from adapter.evaluators.base import BaseEvaluator


@Registry.register("amo_boxed", "evaluator")
class AMOBenchEvaluator(BaseEvaluator):
    """AMO-Bench boxed 答案评估器"""

    def __init__(self, config):
        super().__init__(config)
        self.tolerance = config.get("tolerance", 1e-6)

    def evaluate(self, pred: str, item: DataItem) -> dict:
        """评估模型输出"""
        reference = str(item.reference)
        pred_answer = self._extract_boxed(pred)
        ref_answer = self._extract_boxed(reference)

        if pred_answer is None:
            is_correct = self._compare_answers(pred.strip(), ref_answer or reference)
        else:
            is_correct = self._compare_answers(pred_answer, ref_answer)

        return {"accuracy": 1.0 if is_correct else 0.0}

    def _extract_boxed(self, text: str) -> str:
        """提取 \\boxed{} 中的内容"""
        patterns = [
            r"\\boxed\{([^}]+)\}",
            r"\\boxed\s*\{(.+?)\}",
            r"boxed\{([^}]+)\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _compare_answers(self, pred: str, ref: str) -> bool:
        """比较答案"""
        if pred is None or ref is None:
            return False
        pred_clean = self._clean_answer(pred)
        ref_clean = self._clean_answer(ref)

        try:
            pred_num = float(pred_clean)
            ref_num = float(ref_clean)
            return abs(pred_num - ref_num) < self.tolerance
        except ValueError:
            return pred_clean == ref_clean

    def _clean_answer(self, answer: str) -> str:
        """清理答案"""
        answer = answer.strip()
        answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}',
                       lambda m: str(int(m.group(1)) / int(m.group(2))), answer)
        answer = re.sub(r'\\cdot', '*', answer)
        answer = re.sub(r'\\times', '*', answer)
        answer = re.sub(r'\s+', '', answer)
        return answer


@Registry.register("amo_exact", "evaluator")
class AMOExactEvaluator(BaseEvaluator):
    """AMO 精确匹配评估器"""

    def __init__(self, config):
        super().__init__(config)

    def evaluate(self, pred: str, item: DataItem) -> dict:
        """精确匹配评估"""
        reference = str(item.reference)
        pred_clean = self._normalize(pred)
        ref_clean = self._normalize(reference)
        return {"accuracy": 1.0 if pred_clean == ref_clean else 0.0}

    def _normalize(self, text: str) -> str:
        """规范化文本"""
        return re.sub(r'\s+', '', text.lower().strip())
