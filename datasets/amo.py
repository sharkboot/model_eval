from .custom import CustomDataset
from core.base import DataItem

class AMODataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'amo'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将AMO数据项转换为标准DataItem格式"""
        # AMO的每个数据项包含：问题、答案、类别等
        question = data_item.get('question', '')
        reference = data_item.get('answer', '')
        category = data_item.get('category', '')  # 代数、几何、数论、组合数学等
        problem_type = data_item.get('problem_type', '')  # 选择题、简答题、开放式题目
        year = data_item.get('year', '')
        problem_number = data_item.get('problem_number', '')
        
        # 构建完整的提示词
        full_prompt = f"请解决以下AMO{category}问题，答案应为整数或详细的证明过程。\n\n{question}\n\n答案："
        
        # 构建元数据，包含类别、题目类型、年份和问题编号等信息
        metadata = {
            'category': category,
            'problem_type': problem_type,
            'year': year,
            'problem_number': problem_number,
            'difficulty': data_item.get('difficulty', '')
        }
        
        categories = ['amo']
        if category:
            categories.append(category)
        if problem_type:
            categories.append(problem_type)
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
