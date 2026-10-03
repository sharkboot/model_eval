"""
AIME 2025 适配器

AIME (American Invitational Mathematics Examination) 由 MAA 主办，
每年有两卷：AIME I（11月）与 AIME II（2月），各 15 题。
答案统一为 0-999 的三位整数（无前导零则补 0）。

官方参考:
- AoPS Wiki: https://artofproblemsolving.com/wiki/index.php/2025_AIME_I
- 官方解题: https://live.poshenloh.com/images/past-contests/pdf/aime-2025I-solutions.pdf
"""

import os
import re

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


def _extract_roman(text: str) -> str:
    """从文本中提取罗马数字 (I, II, III...) 返回大写字母。"""
    m = re.search(r'\b([IVXLC]+)\b', text)
    return m.group(1).upper() if m else ""


@Registry.register("AIME2025", "dataset")
class AIME2025Dataset(BaseDataset):
    """
    AIME 2025 数据集适配器

    数据格式:
    - problem/question: 问题描述
    - answer: 答案（整数 0-999）
    - year: 年份（默认 2025）
    - number: 题号（1-15）
    - source/exam: 来源标记（如 "AIME I", "AIME-II"）
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("AIME2025 requires 'data_path' config")
        self.dataset_name = "AIME2025"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = (
            data_item.get('problem')
            or data_item.get('question')
            or data_item.get('prompt', '')
        )
        answer = data_item.get('answer', '')

        # 提取卷次 (I / II)
        source_text = (
            data_item.get('source', '')
            or data_item.get('exam', '')
            or ''
        )
        volume = _extract_roman(source_text) or 'UNKNOWN'

        # 生成唯一 ID：题号 + 卷次
        number = data_item.get('number', '')
        problem_id = str(number).strip() if number else ''
        if not problem_id:
            problem_id = f"{data_item.get('year', 2025) % 100:02d}_unknown"
        else:
            problem_id = f"{problem_id}_{volume}"

        return DataItem(
            id=self.build_id(problem_id),
            prompt=question,
            reference=str(answer).strip(),
            metadata={
                'year': data_item.get('year', 2025),
                'number': str(number).strip() if number else '',
                'volume': volume,
                'source': source_text,
            },
            category=['math', 'competition', 'AIME', volume],
            difficulty='olympiad',
        )


@Registry.register("AIME2025-I", "dataset")
class AIME2025IDataset(AIME2025Dataset):
    """AIME 2025 I 卷子集"""

    def preprocess(self, data_item):
        source_text = (
            data_item.get('source', '')
            or data_item.get('exam', '')
            or ''
        )
        if 'II' in source_text.upper():
            return None
        return super().preprocess(data_item)


@Registry.register("AIME2025-II", "dataset")
class AIME2025IIDataset(AIME2025Dataset):
    """AIME 2025 II 卷子集"""

    def preprocess(self, data_item):
        source_text = (
            data_item.get('source', '')
            or data_item.get('exam', '')
            or ''
        )
        if 'I' in source_text.upper() and 'II' not in source_text.upper():
            return None
        return super().preprocess(data_item)
