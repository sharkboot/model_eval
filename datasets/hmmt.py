from .custom import CustomDataset

class HMMTDataset(CustomDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_type = 'hmmt'
    
    def convert_to_case(self, item):
        """将HMMT数据项转换为标准案例格式"""
        # HMMT的每个数据项包含：问题、答案、类别等
        question = item.get('question', '')
        answer = item.get('answer', '')
        category = item.get('category', '')  # 代数、几何、组合数学等
        round_type = item.get('round_type', '')  # 个人赛、团体赛、抢答赛
        year = item.get('year', '')
        problem_number = item.get('problem_number', '')
        
        # 构建完整的提示词
        full_prompt = f"请解决以下HMMT{category}问题，答案应为整数或分数。\n\n{question}\n\n答案："
        
        # 构建元数据，包含类别、轮次类型、年份和问题编号等信息
        metadata = {
            'category': category,
            'round_type': round_type,
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
            'round_types': self._get_unique_round_types(),
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
    
    def _get_unique_round_types(self):
        """获取数据集中的唯一轮次类型"""
        round_types = set()
        for item in self.data:
            round_type = item.get('round_type')
            if round_type:
                round_types.add(round_type)
        return list(round_types)
    
    def _get_unique_years(self):
        """获取数据集中的唯一年份"""
        years = set()
        for item in self.data:
            year = item.get('year')
            if year:
                years.add(year)
        return list(years)
