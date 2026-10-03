"""
WebArena 适配器

数据集来源: https://github.com/web-arena-x/webarena
论文: WebArena: A Real-World Web Benchmark — https://arxiv.org/abs/2307.13856 (ICLR 2024)

WebArena 包含 812 个真实网页任务（电商/论坛/代码托管/管理系统），
评估模型在真实网页环境中的代理能力。
注意：需浏览器自动化环境（Playwright/Selenium）。
"""

import os

from core.base import DataItem
from core.data_reader import read_file
from core.registry import Registry
from datasets.base import BaseDataset


@Registry.register("WebArena", "dataset")
class WebArenaDataset(BaseDataset):
    """
    WebArena 数据集适配器

    数据格式:
    - task_id: 任务 ID
    - intent: 任务描述
    - domains: 涉及的域名列表
    - setup_commands: 初始化命令
    - eval_config: 评估配置
    """

    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get('data_path')
        if not self.data_path:
            raise ValueError("WebArena requires 'data_path' config")
        self.dataset_name = "WebArena"

    def load_raw_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return read_file(self.data_path)

    def preprocess(self, data_item):
        task_id = data_item.get('task_id', '')
        intent = data_item.get('intent', '')
        domains = data_item.get('domains', [])
        setup_commands = data_item.get('setup_commands', [])

        # 构建 prompt
        prompt = f"""你是一个网页助手，需要完成以下任务：

任务描述: {intent}

涉及的网站: {', '.join(domains) if domains else '多个网页'}

请在浏览器中完成此任务，并将最终结果提供给我。"""

        # WebArena 的特殊性：答案由浏览器环境验证
        reference = ''

        return DataItem(
            id=self.build_id(task_id),
            prompt=prompt,
            reference=reference,
            metadata={
                'task_id': task_id,
                'domains': domains,
                'setup_commands': setup_commands,
                'original_intent': intent,
            },
            category=['web_agent', 'browser', 'real_world'] if domains else ['web_agent'],
            difficulty='extreme',
        )
