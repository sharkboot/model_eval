"""
OlympiadBench 适配器

数据集来源:
- HuggingFace: OpenBMB/OlympiadBench
- GitHub: https://github.com/OpenBMB/OlympiadBench
- 论文: https://arxiv.org/abs/2402.14008

OlympiadBench 是一个 Olympiad 级别的双语数学推理基准，
包含 825 道来自国际数学奥林匹克 (IMO) 级别的问题。
"""

import os

from core.base import DataItem
from core.data_normalizer import normalize_qa_item
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("OlympiadBench", "dataset")
class OlympiadBenchDataset(BaseDataset):
    """OlympiadBench 数据集适配器

    官方 schema (OpenBMB/OlympiadBench):
    - id: 问题 ID
    - subfield: 子领域 (如 Geometry, Algebra)
    - question: 问题描述
    - solution: 解答过程 (列表)
    - final_answer: 最终答案 (列表)
    - is_multiple_answer: 是否多答案
    - answer_type: 答案类型 (Numerical, Short, Boxed)
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("OlympiadBench requires 'data_path' config")
        self.dataset_name = "OlympiadBench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        # 使用归一化层处理标准字段
        normalized = normalize_qa_item(
            data_item,
            q_keys=["question", "problem", "prompt"],
            a_keys=["final_answer", "answer", "solution"],
            c_keys=["subfield", "subject", "category"],
            d_keys=["difficulty", "level"],
        )

        # OlympiadBench 的特殊处理：
        # - solution 是列表，合并为字符串
        # - final_answer 可能是列表，取第一个
        solution = data_item.get("solution", [])
        if isinstance(solution, list):
            solution = "\n\n".join(str(s) for s in solution if s)
        else:
            solution = str(solution) if solution else ""

        final_answer = data_item.get("final_answer", "")
        if isinstance(final_answer, list):
            final_answer = final_answer[0] if final_answer else ""

        return DataItem(
            id=self.build_id(data_item.get("id", data_item.get("index", ""))),
            prompt=normalized["question"],
            reference=str(final_answer).strip(),
            metadata={
                "subfield": normalized["metadata"].get("subfield", ""),
                "answer_type": data_item.get("answer_type", ""),
                "is_multiple_answer": data_item.get("is_multiple_answer", False),
                "solution": solution,
            },
            category=normalized["category"],
            difficulty=normalized["difficulty"],
        )


@Registry.register("IMO", "dataset")
class IMODataset(OlympiadBenchDataset):
    """IMO (International Mathematical Olympiad) 数据集适配器"""

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "IMO"

    def preprocess(self, data_item):
        # IMO 题目通常有 source="IMO"
        data_item["source"] = "IMO"
        return super().preprocess(data_item)
