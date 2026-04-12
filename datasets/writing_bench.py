from .custom import CustomDataset
from utils.data_classes import DataItem

class WritingBenchDataset(CustomDataset):
    def __init__(self, config):
        config['data_type'] = 'writing_bench'
        super().__init__(config)
    
    def preprocess(self, data_item):
        """将WritingBench数据项转换为标准DataItem格式"""
        # WritingBench的每个数据项包含：query（写作任务）、materials（参考材料）、criteria（评估标准）等
        prompt = data_item.get('query', '')
        materials = data_item.get('materials', '')
        criteria = data_item.get('criteria', [])
        
        # 构建完整的提示词，包含写作任务和参考材料
        full_prompt = f"# 写作任务\n{prompt}\n\n# 参考材料\n{materials}"
        
        # 参考答案可以留空，因为WritingBench主要评估写作质量，而不是事实正确性
        reference = ""
        
        # 构建元数据，包含评估标准和其他信息
        metadata = {
            'criteria': criteria,
            'domain': data_item.get('domain', ''),
            'subdomain': data_item.get('subdomain', ''),
            'style': data_item.get('style', ''),
            'format': data_item.get('format', ''),
            'length': data_item.get('length', '')
        }
        
        categories = ['writing_bench']
        if metadata.get('domain'):
            categories.append(metadata['domain'])
        if metadata.get('subdomain'):
            categories.append(metadata['subdomain'])
        
        return DataItem(
            id=str(hash(str(data_item))),
            prompt=full_prompt,
            reference=reference,
            metadata=metadata,
            category=categories
        )
