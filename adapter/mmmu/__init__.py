"""
MMMU 适配器

数据集来源: https://huggingface.co/datasets/MMMU/MMMU
论文: MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark — https://arxiv.org/abs/2311.16502 (CVPR 2024)
官方仓库: https://github.com/VisualBenchmarks/MMMU

MMMU 包含 11,500 题，覆盖 30 个学科 × 30 种任务类型，
评估多模态理解与推理能力。数据含图像输入。
注意：需多模态模型支持图像输入。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("MMMU", "dataset")
class MMMUDataset(BaseDataset):
    """
    MMMU 数据集适配器

    数据格式:
    - question: 问题描述
    - images: 图像列表（路径或 base64）
    - options: 选项列表
    - answer: 正确答案
    - subject: 学科分类
    - task: 任务类型
    """

    INDEX_TO_LETTER = {i: chr(65 + i) for i in range(10)}

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        self.image_dir = config.get('image_dir', 'data/mmmu/images')
        if not self.data_path:
            raise ValueError("MMMU requires 'data_path' config")
        self.dataset_name = "MMMU"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = data_item.get('question', '')
        images = data_item.get('images', [])
        options = data_item.get('options', [])
        answer = data_item.get('answer', '')
        subject = data_item.get('subject', '')
        task = data_item.get('task', '')

        # 构建 prompt（添加图像占位符）
        image_desc = f"[图片: {len(images)} 张]" if images else ""
        prompt = f"{question}\n\n{image_desc}"

        # 添加选项
        if options:
            options_lines = []
            for i, opt in enumerate(options[:10]):
                letter = self.INDEX_TO_LETTER.get(i, chr(65 + i))
                options_lines.append(f"{letter}. {opt}")
            prompt += "\n" + "\n".join(options_lines)

        # 标准化答案
        if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
            ref = answer.upper()
        elif isinstance(answer, int) and 0 <= answer <= 9:
            ref = self.INDEX_TO_LETTER[answer]
        else:
            ref = str(answer).strip()

        return DataItem(
            id=self.build_id(data_item.get('id', '')),
            prompt=prompt,
            reference=ref,
            metadata={
                'images': images,
                'image_paths': [os.path.join(self.image_dir, img) if isinstance(img, str) else img for img in images],
                'subject': subject,
                'task': task,
                'original_question': question,
            },
            category=['multimodal', subject, task] if subject else ['multimodal'],
            difficulty='hard',
        )

    def get_image_paths(self, item) -> list:
        """获取图像路径列表"""
        return item.metadata.get('image_paths', [])
