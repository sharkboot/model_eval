"""
WritingBench 适配器

数据集来源: https://github.com/THUDM/WritingBench
论文: WritingBench: A Comprehensive Benchmark for Evaluating LLM Writing Capabilities

数据格式:
- index: 编号
- domain1: 一级领域 (Finance & Business, Politics & Law, Literature & Arts, etc.)
- domain2: 二级领域 (Paper Outline, News Writing, etc.)
- lang: 语言 (zh/en)
- query: 写作任务描述
- checklist: 评估标准列表，每项包含:
    - name: 标准名称
    - criteria_description: 标准描述
    - 1-2, 3-4, 5-6, 7-8, 9-10: 不同分数段的评分规则
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("WritingBench", "dataset")
class WritingBenchDataset(BaseDataset):
    """
    WritingBench 数据集适配器

    评估大语言模型在多个领域（金融、商业、法律、文学、艺术等）
    的中英文写作能力。

    注意: WritingBench 使用 LLM-as-Judge 进行评估，
    需要专门的评估器来比较模型输出与评估标准。
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("WritingBench requires 'data_path' config")
        self.dataset_name = "WritingBench"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        return read_file(self.data_path)

    def preprocess(self, data_item):
        """
        预处理 WritingBench 数据

        WritingBench 的评估使用 LLM-as-Judge 方法，
        需要专门的评估器来评估写作质量。
        """
        query = data_item.get('query', '')
        checklist = data_item.get('checklist', [])

        # 构建评估标准摘要
        criteria_summary = []
        if isinstance(checklist, list):
            for i, item in enumerate(checklist, 1):
                if isinstance(item, dict):
                    name = item.get('name', f'criterion_{i}')
                    desc = item.get('criteria_description', '')
                    criteria_summary.append(f"{i}. {name}: {desc}")

        return DataItem(
            id=self.build_id(data_item.get('index', '')),
            prompt=query,
            reference=checklist,  # 参考答案为完整的评估标准列表
            metadata={
                'index': data_item.get('index'),
                'domain1': data_item.get('domain1'),
                'domain2': data_item.get('domain2'),
                'lang': data_item.get('lang'),
                'checklist': checklist,  # 原始评估标准
                'criteria_summary': '\n'.join(criteria_summary) if criteria_summary else ''
            },
            category=[
                data_item.get('domain1'),
                data_item.get('domain2'),
                data_item.get('lang')
            ]
        )
