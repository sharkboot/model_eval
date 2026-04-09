from .custom import CustomDataset

class IMODataset(CustomDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_type = 'imo'
    
    def convert_to_case(self, item):
        """将IMO数据项转换为标准案例格式"""
        # IMO的每个数据项包含：问题、答案、类别等
        question = item.get('question', '')
        answer = item.get('answer', '')
        category = item.get('category', '')  # 代数、几何、数论、组合数学等
        year = item.get('year', '')
        problem_number = item.get('problem_number', '')
        country = item.get('country', '')  # 题目来源国家
        
        # 构建完整的提示词
        full_prompt = f"请解决以下IMO{category}问题，答案应为详细的证明过程。\n\n{question}\n\n答案："
        
        # 构建元数据，包含类别、年份、问题编号和国家等信息
        metadata = {
            'category': category,
            'year': year,
            'problem_number': problem_number,
            'country': country,
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
            'years': self._get_unique_years(),
            'countries': self._get_unique_countries()
        }
    
    def _get_unique_categories(self):
        """获取数据集中的唯一类别"""
        categories = set()
        for item in self.data:
            category = item.get('category')
            if category:
                categories.add(category)
        return list(categories)
    
    def _get_unique_years(self):
        """获取数据集中的唯一年份"""
        years = set()
        for item in self.data:
            year = item.get('year')
            if year:
                years.add(year)
        return list(years)
    
    def _get_unique_countries(self):
        """获取数据集中的唯一国家"""
        countries = set()
        for item in self.data:
            country = item.get('country')
            if country:
                countries.add(country)
        return list(countries)
