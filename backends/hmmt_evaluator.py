from .base import BaseBackend

class HMMTEvaluator(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'math'
        self.eval_type = config.get('eval_type', 'rule')  # 对于HMMT，主要使用规则评测
    
    def execute(self, model, case, response=None):
        """执行HMMT案例评估
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
        """异步执行HMMT案例评估"""
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
        HMMT要求答案为整数或分数
        """
        import re
        # 提取答案
        # 匹配整数或分数
        match = re.search(r'\b(\d+|\d+/\d+)\b', response)
        if match:
            return match.group(1)
        
        return ""
    
    def get_backend_info(self):
        return {
            'backend_type': 'hmmt',
            'task_type': self.task_type
        }
