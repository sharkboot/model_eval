from .custom import CustomDataset

class CEvalDataset(CustomDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_type = 'ceval'
    
    def convert_to_case(self, item):
        """将C-Eval数据项转换为标准案例格式"""
        # C-Eval的每个数据项包含：问题、选项、答案等
        question = item.get('question', '')
        options = item.get('options', [])
        answer = item.get('answer', '')
        subject = item.get('subject', '')
        
        # 构建完整的提示词，包含问题和选项
        options_text = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
        full_prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n{question}\n{options_text}\n\n答案："
        
        # 构建元数据，包含学科和其他信息
        metadata = {
            'subject': subject,
            'difficulty': item.get('difficulty', ''),
            'category': item.get('category', '')
        }
        
        return {
            'prompt': full_prompt,
            'answer': answer,
            'metadata': metadata
        }
    
    def get_dataset_info(self):
        return {
            'dataset_type': self.dataset_type,
            'num_samples': len(self.data),
            'subjects': self._get_unique_subjects(),
            'categories': self._get_unique_categories()
        }
    
    def _get_unique_subjects(self):
        """获取数据集中的唯一学科"""
        subjects = set()
        for item in self.data:
            subject = item.get('subject')
            if subject:
                subjects.add(subject)
        return list(subjects)
    
    def _get_unique_categories(self):
        """获取数据集中的唯一分类"""
        categories = set()
        for item in self.data:
            category = item.get('category')
            if category:
                categories.add(category)
        return list(categories)
