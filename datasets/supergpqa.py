from .custom import CustomDataset
from core.base import DataItem

class SuperGPQADataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'supergpqa'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将SUPERGPQA数据项转换为标准DataItem格式"""
        # SUPERGPQA的每个数据项包含：问题、选项、答案、学科等
        question = data_item.get('question', '')
        options = data_item.get('options', [])
        reference = data_item.get('answer', '')
        subject = data_item.get('subject', '')
        field = data_item.get('field', '')
        discipline = data_item.get('discipline', '')
        
        # 构建完整的提示词，包含问题和选项
        options_text = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
        full_prompt = f"以下是{discipline}学科的研究生级别问题，请选出正确答案。\n\n{question}\n{options_text}\n\n答案："
        
        # 构建元数据，包含学科、领域和其他信息
        metadata = {
            'subject': subject,
            'field': field,
            'discipline': discipline,
            'difficulty': data_item.get('difficulty', '')
        }
        
        categories = ['supergpqa']
        if discipline:
            categories.append(discipline)
        if field:
            categories.append(field)
        if subject:
            categories.append(subject)
        
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=full_prompt,
            reference=reference,
            metadata=metadata,
            category=categories,
            difficulty=metadata.get('difficulty')
        )
