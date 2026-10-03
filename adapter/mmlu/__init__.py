"""
MMLU 适配器

数据集来源: https://huggingface.co/datasets/EleutherAI/mmlu
论文: MMLU: Measuring Massive Multitask Language Understanding — https://arxiv.org/abs/2009.03300
官方仓库: https://github.com/EleutherAI/lm-evaluation-harness

MMLU 包含 14,269 道选择题，覆盖 57 个学科，
评估大语言模型的多任务知识理解能力。
"""

import os

from core.base import DataItem
from core.data_normalizer import normalize_qa_item
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("MMLU", "dataset")
class MMLUDataset(BaseDataset):
    """
    MMLU 数据集适配器

    数据格式:
    - question: 问题描述
    - choices: 选项列表（4 个字符串）
    - answer: 正确答案索引（0-3，对应 A-D）
    - subject: 学科分类
    """

    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("MMLU requires 'data_path' config")
        self.dataset_name = "MMLU"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        # 使用归一化层提取标准字段
        normalized = normalize_qa_item(
            data_item,
            q_keys=["question", "query"],
            a_keys=["answer", "correct_answer"],
            c_keys=["subject", "category"],
        )

        # MMLU 选项为 list，合并到 prompt
        choices = data_item.get('choices', [])
        if not choices:
            choices = [
                data_item.get('choice_A'),
                data_item.get('choice_B'),
                data_item.get('choice_C'),
                data_item.get('choice_D'),
            ]
            choices = [c for c in choices if c]

        question_text = normalized["question"]
        if choices:
            options_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices[:4])]
            question_text = f"{question_text}\n" + "\n".join(options_lines)

        # 答案索引转字母
        answer_idx = data_item.get('answer', data_item.get('correct_answer', -1))
        if isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
            reference = self.INDEX_TO_LETTER[answer_idx]
        else:
            reference = str(normalized["answer"]).strip()

        return DataItem(
            id=self.build_id(data_item.get('question', '')),
            prompt=question_text,
            reference=reference,
            metadata={
                'subject': normalized["metadata"].get("subject", ""),
                'choices': choices,
            },
            category=[normalized["category"][0] if normalized["category"] else "unknown"],
        )
