from .base import BaseBackend

class CEvalEvaluator(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'mcq'
        self.eval_type = config.get('eval_type', 'rule')  # 对于C-Eval，主要使用规则评测
    
    def execute(self, model, case, response=None):
        """执行C-Eval案例评估
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
        """异步执行C-Eval案例评估"""
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
        C-Eval要求答案格式为："答案：[LETTER]"
        """
        import re
        # 提取答案字母
        match = re.search(r'答案[：:]\s*([A-D])', response)
        if match:
            return match.group(1).upper()
        
        # 如果没有找到标准格式，尝试提取单独的字母
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1).upper()
        
        return ""
    
    def get_backend_info(self):
        return {
            'backend_type': 'ceval',
            'task_type': self.task_type
        }
