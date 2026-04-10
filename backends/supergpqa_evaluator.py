from .base import BaseBackend

class SuperGPQAEvaluator(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'mcq'
        self.eval_type = config.get('eval_type', 'rule')  # 对于SUPERGPQA，主要使用规则评测
    
    def execute(self, model, case, response=None):
        """执行SUPERGPQA案例评估
        输入：案例dict，包含prompt、answer、metadata字段
        输出：评估分数（1.0表示正确，0.0表示错误）
        """
        try:
            if not response:
                return 0.0
            
            # 提取模型预测的答案
            predicted_answer = self._extract_answer(response)
            
            # 与参考答案比较
            if predicted_answer == case['answer']:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    async def async_execute(self, model, case, response=None):
        """异步执行SUPERGPQA案例评估"""
        try:
            if not response:
                return 0.0
            
            # 提取模型预测的答案
            predicted_answer = self._extract_answer(response)
            
            # 与参考答案比较
            if predicted_answer == case['answer']:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def _extract_answer(self, response):
        """从模型响应中提取答案
        SUPERGPQA是多选题，答案为A、B、C、D等
        """
        import re
        # 提取答案
        # 匹配A、B、C、D等选项
        match = re.search(r'\b([ABCD])\b', response)
        if match:
            return match.group(1)
        
        # 匹配答案：A这样的格式
        match = re.search(r'答案[:：]\s*([ABCD])', response)
        if match:
            return match.group(1)
        
        return ""
    
    def get_backend_info(self):
        return {
            'backend_type': 'supergpqa',
            'task_type': self.task_type
        }
