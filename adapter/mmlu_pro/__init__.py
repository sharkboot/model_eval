"""
MMLU-Pro 适配器

数据集来源: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
论文: MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark — https://arxiv.org/abs/2406.01574 (NeurIPS 2024 D&B)
官方仓库: https://github.com/TIGER-AI-Lab/MMLU-Pro

MMLU-Pro 包含 12,000 道题，覆盖 14 个领域，
选项从 MMLU 的 4 个增至 10 个，显著提升区分度。
"""

import os

from core.base import DataItem
from core.data_normalizer import normalize_qa_item
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("MMLU-Pro", "dataset")
class MMLUProDataset(BaseDataset):
    """
    MMLU-Pro 数据集适配器

    数据格式:
    - question: 问题描述
    - options: 选项列表（10 个字符串）
    - answer: 正确答案索引（0-9）
    - subject: 学科分类
    - category: 领域分类
    - explanation: 解析
    """

    INDEX_TO_LETTER = {i: chr(65 + i) for i in range(10)}  # A-J

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("MMLU-Pro requires 'data_path' config")
        self.dataset_name = "MMLU-Pro"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        normalized = normalize_qa_item(
            data_item,
            q_keys=["question", "query"],
            a_keys=["answer", "correct_answer"],
            c_keys=["subject", "category"],
        )

        # options 为 list
        options = data_item.get('options', [])
        if not options:
            # 尝试 A-J 键
            options = [data_item.get(chr(65 + i)) for i in range(10)]
            options = [opt for opt in options if opt]

        question_text = normalized["question"]
        if options:
            options_lines = [f"{self.INDEX_TO_LETTER[i]}. {opt}" for i, opt in enumerate(options[:10])]
            question_text = f"{question_text}\n" + "\n".join(options_lines)

        # 答案索引转字母
        answer_idx = data_item.get('answer', data_item.get('correct_answer', -1))
        if isinstance(answer_idx, int) and 0 <= answer_idx <= 9:
            reference = self.INDEX_TO_LETTER[answer_idx]
        else:
            reference = str(normalized["answer"]).strip()

        return DataItem(
            id=self.build_id(data_item.get('question', '')),
            prompt=question_text,
            reference=reference,
            metadata={
                'subject': normalized["metadata"].get("subject", ""),
                'category': normalized["metadata"].get("category", ""),
                'explanation': data_item.get('explanation', ''),
                'options': options,
            },
            category=[normalized["category"][0] if normalized["category"] else "unknown"],
        )
