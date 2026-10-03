"""
EqBench 适配器

WARNING: 此适配器为占位实现。

原始声称来源 "EqBench: A Mathematical Reasoning Benchmark" 无法查证。
GitHub https://github.com/EqBench/EqBench 实际指向程序等价性相关组织，
而非数学推理基准。已知同名公开数据集为 Badihi, Li, Rubin (MSR 2021) 的
[程序等价检查基准](https://github.com/shrBadihi/EqBench)，字段为
Benchmark name / Program name / LOC / # loops 等，与 question/answer
结构完全不符。

如后续获得真实数学推理数据集，请更新 docstring、字段映射及难度阈值。
"""

import os
import logging

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset

logger = logging.getLogger(__name__)
logger.warning(
    "[EqBench] 此适配器为占位实现，数据来源存疑。"
    "请核实真实数据集后更新 preprocess() 和难度阈值。"
    "已知同名数据集为程序等价性基准 (shrBadihi/EqBench)。"
)


@Registry.register("EqBench", "dataset", allow_override=True)
class EqBenchDataset(BaseDataset):
    """EqBench 数据集适配器（占位实现）"""

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("EqBench requires 'data_path' config")
        self.dataset_name = "EqBench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        # TODO: 待核实真实数据集 schema 后替换此逻辑
        question = (
            data_item.get('question')
            or data_item.get('problem')
            or data_item.get('prompt', '')
        )
        answer = (
            data_item.get('answer')
            or data_item.get('solution', '')
        )
        difficulty = data_item.get('difficulty', 5)
        return DataItem(
            id=self.build_id(data_item.get('id', data_item.get('question_id', ''))),
            prompt=question,
            reference=str(answer),
            metadata={
                'difficulty': difficulty,
                'type': data_item.get('type', ''),
                'warning': 'placeholder_impl',
            },
            category=['math', 'reasoning', data_item.get('type', '')],
            difficulty='unknown',
        )
