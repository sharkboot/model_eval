from .custom import CustomDataset
from core.base import DataItem

class EQBenchDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'eq_bench'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将EQ-Bench数据项转换为标准DataItem格式"""
        # EQ-Bench的每个数据项包含：场景、情绪类型、参考评分等
        scenario = data_item.get('scenario', '')
        emotions = data_item.get('emotions', [])
        reference_ratings = data_item.get('reference_ratings', {})
        context = data_item.get('context', '')
        
        # 构建完整的提示词
        emotions_text = "\n".join([f"- {emotion}" for emotion in emotions])
        format_example = "\n".join([f"{emotion}: {reference_ratings.get(emotion, 0)}" for emotion in emotions])
        full_prompt = f"请根据以下场景，对每种情绪的强度进行评分（0-10分，0表示完全没有，10表示非常强烈）。\n\n场景：{scenario}\n\n需要评分的情绪：\n{emotions_text}\n\n请按照以下格式输出评分结果：\n{format_example}\n"
        
        # 构建元数据，包含场景、情绪类型和其他信息
        metadata = {
            'scenario': scenario,
            'emotions': emotions,
            'reference_ratings': reference_ratings,
            'context': context
        }
        
        categories = ['eq_bench']
        categories.extend(emotions)
        # 简单分类场景类型
        if '工作' in scenario:
            categories.append('工作')
        elif '家庭' in scenario:
            categories.append('家庭')
        elif '朋友' in scenario:
            categories.append('朋友')
        elif '社交' in scenario:
            categories.append('社交')
        else:
            categories.append('其他')
        
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=full_prompt,
            reference=reference_ratings,
            metadata=metadata,
            category=categories
        )
