"""
MATH-Vista 适配器

数据集来源: https://huggingface.co/datasets/LMMs-Lab/MathVista
论文: MATH-Vista: Evaluating Math Reasoning in Visual Contexts — https://arxiv.org/abs/2310.02255 (ICLR 2024)
官网: https://mathvista.github.io/

MATH-Vista 包含 3,948 题，评估模型在视觉环境中的数学推理能力，
数据包含图表、几何图形等可视化数学内容。
注意：需多模态模型支持图像输入。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("MATH-Vista", "dataset")
class MATHVistaDataset(BaseDataset):
    """
    MATH-Vista 数据集适配器

    数据格式:
    - question: 问题描述
    - images: 图像路径列表
    - answer: 答案（可能含 \\boxed{}）
    - task: 任务类型
    - category: 类别
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        self.image_dir = config.get('image_dir', 'data/mathvista/images')
        if not self.data_path:
            raise ValueError("MATH-Vista requires 'data_path' config")
        self.dataset_name = "MATH-Vista"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        question = data_item.get('question', '')
        images = data_item.get('images', [])
        answer = data_item.get('answer', '')
        task = data_item.get('task', '')
        category = data_item.get('category', '')

        # 构建 prompt
        image_desc = f"[图片: {len(images)} 张]" if images else ""
        prompt = f"{question}\n\n{image_desc}"

        # 标准化答案（提取 boxed）
        import re
        m = re.search(r'\\boxed\{([^}]+)\}', str(answer))
        reference = m.group(1).strip() if m else str(answer).strip()

        return DataItem(
            id=self.build_id(data_item.get('pid', data_item.get('id', ''))),
            prompt=prompt,
            reference=str(reference),
            metadata={
                'images': images,
                'image_paths': [os.path.join(self.image_dir, img) if isinstance(img, str) else img for img in images],
                'task': task,
                'category': category,
                'original_answer': answer,
            },
            category=['multimodal', 'math', task, category] if task else ['multimodal', 'math'],
            difficulty='hard',
        )

    def get_image_paths(self, item) -> list:
        """获取图像路径列表"""
        return item.metadata.get('image_paths', [])
