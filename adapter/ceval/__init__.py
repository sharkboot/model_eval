"""
C-Eval 适配器

数据集来源: https://huggingface.co/datasets/ceval/ceval-exam
论文: C-Eval: A Multi-Level Multi-Domain Chinese Evaluation Benchmark

C-Eval 是一个中文大语言模型评估基准，包含13948道选择题，
涵盖4个难度级别和52个学科。
"""

import os

from core.base import DataItem
from core.data_normalizer import normalize_qa_item, extract_options
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("CEval", "dataset")
class CEvalDataset(BaseDataset):
    """
    C-Eval 数据集适配器

    数据格式 (官方 schema):
    - id: 问题ID
    - exam: 考试类型 (cloud_computing, advanced_mathematics, ...)
    - question: 问题描述
    - A/B/C/D: 选项
    - answer: 答案 (A/B/C/D)
    - explanation: 解析
    - category: 学科类别
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("CEval requires 'data_path' config")
        self.dataset_name = "CEval"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        # 使用归一化层提取标准字段
        normalized = normalize_qa_item(
            data_item,
            q_keys=["question", "problem", "prompt"],
            a_keys=["answer", "reference", "solution"],
            c_keys=["category", "exam", "subject"],
        )

        # C-Eval 特殊处理：将选项合并到问题中
        options = extract_options(data_item)
        if options:
            normalized["question"] = f"{normalized['question']}\n" + "\n".join(options)

        # 提取解析
        explanation = data_item.get('explanation', '')

        return DataItem(
            id=self.build_id(data_item.get('id', '')),
            prompt=normalized["question"],
            reference=normalized["answer"],
            metadata={
                'explanation': explanation,
                'exam': normalized["metadata"].get("exam", ""),
            },
            category=normalized["category"],
        )


@Registry.register("CEvalHard", "dataset")
class CEvalHardDataset(CEvalDataset):
    """
    C-Eval 高难度子集

    注意：C-Eval 官方不提供 difficulty 字段。
    本适配器通过 exam 类型筛选高难度题目（如高等数学、物理等）。
    """

    # 高难度 exam 类型列表
    HARD_EXAMS = {
        "advanced_mathematics", "physics", "probability_and_statistics",
        "discrete_mathematics", "electrical_engineer", "computer_sience",
        "professional_teachers",
    }

    def preprocess(self, data_item):
        exam = data_item.get('exam', '')
        if exam not in self.HARD_EXAMS:
            return None
        return super().preprocess(data_item)
