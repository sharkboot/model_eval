"""
TruthfulQA 适配器

数据集来源: https://huggingface.co/datasets/truthfulqa/truthful_qa
论文: TruthfulQA: Measuring How Models Mimic Human Falsehoods — https://arxiv.org/abs/2109.07958 (NeurIPS 2021)
官方仓库: https://github.com/llm-testing/truthfulqa

TruthfulQA 包含 817 题，80 个主题，
测试模型在常见错误认知和信念上的诚实性。
需要 LLM 判分（best vs other 语义匹配）。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("TruthfulQA", "dataset")
class TruthfulQADataset(BaseDataset):
    """
    TruthfulQA 数据集适配器

    数据格式:
    - question: 问题
    - best_answer: 最佳回答
    - other_answers: 其他可接受回答
    - category: 错误认知类别
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("TruthfulQA requires 'data_path' config")
        self.dataset_name = "TruthfulQA"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = data_item.get('question', '')
        best_answer = data_item.get('best_answer', '')
        other_answers = data_item.get('other_answers', [])
        category = data_item.get('category', '')

        # TruthfulQA 的特殊性：没有单一正确答案
        # 参考回答是多条可能正确的回答
        reference = best_answer
        if other_answers:
            reference = f"{best_answer}\n\n或：\n" + "\n\n或：\n".join(str(a) for a in other_answers[:3])

        return DataItem(
            id=self.build_id(question),
            prompt=question,
            reference=reference,
            metadata={
                'best_answer': best_answer,
                'other_answers': other_answers,
                'category': category,
            },
            category=['truthfulness', category] if category else ['truthfulness'],
            difficulty='medium',
        )


# TruthfulQA 专用评估器 — 需要 LLM 判分
@Registry.register("truthfulqa_judge", "evaluator")
class TruthfulQAJudgeEvaluator:
    """
    TruthfulQA LLM-as-Judge 评估器

    使用 LLM 判断模型回答是否与参考答案语义一致。
    需要配置 judge_model 和 judge_api_config。
    """

    def __init__(self, config):
        self.config = config
        self.judge_model = config.get('judge_model', 'claude')
        self.judge_api_config = config.get('judge_api_config', {})

    def evaluate(self, pred: str, item) -> dict:
        """评估回答的真实性。

        Returns:
            {"accuracy": 0.0|1.0, "explanation": str}
        """
        # TODO: 实现真实 LLM 判分逻辑
        # 当前为占位实现
        return {
            "accuracy": 0.5,  # 占位值
            "explanation": "LLM-as-Judge 评估需要配置 judge_model 和 API",
            "category": item.metadata.get('category', ''),
        }
