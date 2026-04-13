from .base import BaseEvaluator
from core.base import EvaluationResult

class IMOMEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'math'
        self.eval_type = config.get('eval_type', 'rule')  # 对于IMO，主要使用规则评测
    
    def evaluate(self, data_item, model_output):
        """执行IMO案例评估
        输入：DataItem和模型输出
        输出：EvaluationResult
        """
        try:
            if not model_output:
                score = 0.0
            else:
                # 提取模型预测的答案
                predicted_answer = self._extract_answer(model_output)
                
                # 与参考答案比较
                if predicted_answer == str(data_item.reference):
                    score = 1.0
                else:
                    score = 0.0
        except Exception as e:
            score = 0.0
        
        metrics = {'score': score, 'accuracy': score}
        
        return EvaluationResult(
            data_id=data_item.id,
            evaluator_name='IMOMEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
    def _extract_answer(self, response):
        """从模型响应中提取答案
        IMO要求答案为详细的证明过程，这里简化处理，只提取关键结论
        """
        import re
        # 提取答案
        # 匹配整数
        match = re.search(r'\b(\d+)\b', response)
        if match:
            return match.group(1)
        
        # 匹配关键结论
        keywords = ['证明', '结论', '答案', 'result', 'proof', 'conclusion']
        for keyword in keywords:
            if keyword in response:
                # 提取关键词后的内容
                index = response.find(keyword)
                if index != -1:
                    return response[index:index+100].strip()
        
        return ""

