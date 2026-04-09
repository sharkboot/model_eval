from .custom import CustomDataset

class AMODataset(CustomDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_type = 'amo'
    
    def convert_to_case(self, item):
        """将AMO数据项转换为标准案例格式"""
        # AMO的每个数据项包含：问题、答案、类别等
        question = item.get('question', '')
        answer = item.get('answer', '')
        category = item.get('category', '')  # 代数、几何、数论、组合数学等
        problem_type = item.get('problem_type', '')  # 选择题、简答题、开放式题目
        year = item.get('year', '')
        problem_number = item.get('problem_number', '')
        
        # 构建完整的提示词
        full_prompt = f"请解决以下AMO{category}问题，答案应为整数或详细的证明过程。\n\n{question}\n\n答案："
        
        # 构建元数据，包含类别、题目类型、年份和问题编号等信息
        metadata = {
            'category': category,
            'problem_type': problem_type,
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
            'categories': self._get_unique_categories(),
            'problem_types': self._get_unique_problem_types(),
            'years': self._get_unique_years()
        }
    
    def _get_unique_categories(self):
        """获取数据集中的唯一类别"""
        categories = set()
        for item in self.data:
            category = item.get('category')
            if category:
                categories.add(category)
        return list(categories)
    
    def _get_unique_problem_types(self):
        """获取数据集中的唯一题目类型"""
        problem_types = set()
        for item in self.data:
            problem_type = item.get('problem_type')
            if problem_type:
                problem_types.add(problem_type)
        return list(problem_types)
    
    def _get_unique_years(self):
        """获取数据集中的唯一年份"""
        years = set()
        for item in self.data:
            year = item.get('year')
            if year:
                years.add(year)
        return list(years)
