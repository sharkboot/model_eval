from .custom import CustomDataset
from utils.data_classes import DataItem

class IMODataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'imo'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将IMO数据项转换为标准DataItem格式"""
        # IMO的每个数据项包含：问题、答案、类别等
        question = data_item.get('question', '')
        reference = data_item.get('answer', '')
        category = data_item.get('category', '')  # 代数、几何、数论、组合数学等
        year = data_item.get('year', '')
        problem_number = data_item.get('problem_number', '')
        country = data_item.get('country', '')  # 题目来源国家
        
        # 构建完整的提示词
        full_prompt = f"请解决以下IMO{category}问题，答案应为详细的证明过程。\n\n{question}\n\n答案："
        
        # 构建元数据，包含类别、年份、问题编号和国家等信息
        metadata = {
            'category': category,
            'year': year,
            'problem_number': problem_number,
            'country': country,
            'difficulty': data_item.get('difficulty', '')
        }
        
        categories = ['imo']
        if category:
            categories.append(category)
        if year:
            categories.append(f'year_{year}')
        if country:
            categories.append(country)
        
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=full_prompt,
            reference=reference,
            metadata=metadata,
            category=categories,
            difficulty=metadata.get('difficulty')
        )
