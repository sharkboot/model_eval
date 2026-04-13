from .custom import CustomDataset
from core.base import DataItem

class CEvalDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'ceval'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将C-Eval数据项转换为标准DataItem格式"""
        # C-Eval的每个数据项包含：问题、选项、答案等
        question = data_item.get('question', '')
        options = data_item.get('options', [])
        reference = data_item.get('answer', '')
        subject = data_item.get('subject', '')
        
        # 构建完整的提示词，包含问题和选项
        options_text = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
        full_prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n{question}\n{options_text}\n\n答案："
        
        # 构建元数据，包含学科和其他信息
        metadata = {
            'subject': subject,
            'difficulty': data_item.get('difficulty', ''),
            'category': data_item.get('category', '')
        }
        
        categories = ['ceval']
        if subject:
            categories.append(subject)
        if metadata.get('category'):
            categories.append(metadata['category'])
        
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=full_prompt,
            reference=reference,
            metadata=metadata,
            category=categories,
            difficulty=metadata.get('difficulty')
        )
