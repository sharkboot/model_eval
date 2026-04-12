from .custom import CustomDataset
from utils.data_classes import DataItem

class AIMEDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'aime'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将AIME数据项转换为标准DataItem格式"""
        # AIME的每个数据项包含：问题、答案等
        question = data_item.get('question', '')
        reference = data_item.get('answer', '')
        year = data_item.get('year', '')
        problem_number = data_item.get('problem_number', '')
        
        # 构建完整的提示词
        full_prompt = f"请解决以下AIME问题，答案应为0-999之间的整数。\n\n{question}\n\n答案："
        
        # 构建元数据，包含年份和问题编号等信息
        metadata = {
            'year': year,
            'problem_number': problem_number,
            'difficulty': data_item.get('difficulty', '')
        }
        
        categories = ['aime']
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
