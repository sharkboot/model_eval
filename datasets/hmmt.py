from .custom import CustomDataset
from core.base import DataItem

class HMMTDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'hmmt'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将HMMT数据项转换为标准DataItem格式"""
        # HMMT的每个数据项包含：问题、答案、类别等
        question = data_item.get('question', '')
        reference = data_item.get('answer', '')
        category = data_item.get('category', '')  # 代数、几何、组合数学等
        round_type = data_item.get('round_type', '')  # 个人赛、团体赛、抢答赛
        year = data_item.get('year', '')
        problem_number = data_item.get('problem_number', '')
        
        # 构建完整的提示词
        full_prompt = f"请解决以下HMMT{category}问题，答案应为整数或分数。\n\n{question}\n\n答案："
        
        # 构建元数据，包含类别、轮次类型、年份和问题编号等信息
        metadata = {
            'category': category,
            'round_type': round_type,
            'year': year,
            'problem_number': problem_number,
            'difficulty': data_item.get('difficulty', '')
        }
        
        categories = ['hmmt']
        if category:
            categories.append(category)
        if round_type:
            categories.append(round_type)
        if year:
            categories.append(f'year_{year}')
        
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=full_prompt,
            reference=reference,
            metadata=metadata,
            category=categories,
            difficulty=metadata.get('difficulty')
        )
