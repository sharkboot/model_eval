from .custom import CustomDataset

class WritingBenchDataset(CustomDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_type = 'writing_bench'
    
    def convert_to_case(self, item):
        """将WritingBench数据项转换为标准案例格式"""
        # WritingBench的每个数据项包含：query（写作任务）、materials（参考材料）、criteria（评估标准）等
        prompt = item.get('query', '')
        materials = item.get('materials', '')
        criteria = item.get('criteria', [])
        
        # 构建完整的提示词，包含写作任务和参考材料
        full_prompt = f"# 写作任务\n{prompt}\n\n# 参考材料\n{materials}"
        
        # 参考答案可以留空，因为WritingBench主要评估写作质量，而不是事实正确性
        answer = ""
        
        # 构建元数据，包含评估标准和其他信息
        metadata = {
            'criteria': criteria,
            'domain': item.get('domain', ''),
            'subdomain': item.get('subdomain', ''),
            'style': item.get('style', ''),
            'format': item.get('format', ''),
            'length': item.get('length', '')
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
            'domains': self._get_unique_domains(),
            'subdomains': self._get_unique_subdomains()
        }
    
    def _get_unique_domains(self):
        """获取数据集中的唯一领域"""
        domains = set()
        for item in self.data:
            domain = item.get('domain')
            if domain:
                domains.add(domain)
        return list(domains)
    
    def _get_unique_subdomains(self):
        """获取数据集中的唯一子领域"""
        subdomains = set()
        for item in self.data:
            subdomain = item.get('subdomain')
            if subdomain:
                subdomains.add(subdomain)
        return list(subdomains)
