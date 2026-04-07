from .base import BaseBackend

class AIMEEvaluator(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'math'
        self.eval_type = config.get('eval_type', 'rule')  # 对于AIME，主要使用规则评测
    
    def execute(self, model, case, response=None):
        """执行AIME案例评估
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
        """异步执行AIME案例评估"""
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
        AIME要求答案为0-999之间的整数
        """
        import re
        # 提取数字
        match = re.search(r'\b(\d{1,3})\b', response)
        if match:
            return match.group(1)
        
        return ""
    
    def get_backend_info(self):
        return {
            'backend_type': 'aime',
            'task_type': self.task_type
        }
