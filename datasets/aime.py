from .custom import CustomDataset

class AIMEDataset(CustomDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_type = 'aime'
    
    def convert_to_case(self, item):
        """将AIME数据项转换为标准案例格式"""
        # AIME的每个数据项包含：问题、答案等
        question = item.get('question', '')
        answer = item.get('answer', '')
        year = item.get('year', '')
        problem_number = item.get('problem_number', '')
        
        # 构建完整的提示词
        full_prompt = f"请解决以下AIME问题，答案应为0-999之间的整数。\n\n{question}\n\n答案："
        
        # 构建元数据，包含年份和问题编号等信息
        metadata = {
            'year': year,
            'problem_number': problem_number,
            'difficulty': item.get('difficulty', '')
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
            'years': self._get_unique_years(),
            'problem_numbers': self._get_unique_problem_numbers()
        }
    
    def _get_unique_years(self):
        """获取数据集中的唯一年份"""
        years = set()
        for item in self.data:
            year = item.get('year')
            if year:
                years.add(year)
        return list(years)
    
    def _get_unique_problem_numbers(self):
        """获取数据集中的唯一问题编号"""
        problem_numbers = set()
        for item in self.data:
            problem_number = item.get('problem_number')
            if problem_number:
                problem_numbers.add(problem_number)
        return list(problem_numbers)
