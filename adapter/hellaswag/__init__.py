"""
HellaSwag 适配器

数据集来源: https://huggingface.co/datasets/Rowan/hellaswag
论文: HellaSwag: Can a Machine Really Finish Your Sentence? — https://arxiv.org/abs/1905.07830 (EMNLP 2019)
官方仓库: https://github.com/rowanz/hellaswag

HellaSwag 包含 10,042 题生活常识句子补全，
评估模型在上下文理解方面的能力。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("HellaSwag", "dataset")
class HellaSwagDataset(BaseDataset):
    """
    HellaSwag 数据集适配器

    数据格式:
    - ctx: 上下文（句子开头）
    - endings: 4 个候选结尾
    - labelId: 正确答案索引（0-3）
    - activity_label: 活动类型
    """

    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("HellaSwag requires 'data_path' config")
        self.dataset_name = "HellaSwag"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        ctx = data_item.get('ctx', '')
        endings = data_item.get('endings', [])
        label_id = data_item.get('labelId', -1)
        activity = data_item.get('activity_label', '')

        # 构建完整问题
        prompt = ctx
        if endings:
            options_lines = [f"{self.INDEX_TO_LETTER[i]}. {e}" for i, e in enumerate(endings[:4])]
            prompt = f"{ctx}\n\n{chr(10)}".join(options_lines)

        # 答案转字母
        if isinstance(label_id, int) and 0 <= label_id <= 3:
            reference = self.INDEX_TO_LETTER[label_id]
        else:
            reference = ""

        return DataItem(
            id=self.build_id(data_item.get('ind', '')),
            prompt=prompt,
            reference=reference,
            metadata={
                'ctx_full': ctx,
                'endings': endings,
                'activity': activity,
            },
            category=['commonsense', 'completion', activity] if activity else ['commonsense', 'completion'],
            difficulty='easy',
        )
