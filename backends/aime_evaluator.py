from .base import BaseEvaluator
from utils.data_classes import EvaluationResult

class AIMEEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'math'
        self.eval_type = config.get('eval_type', 'rule')  # 对于AIME，主要使用规则评测
    
    def evaluate(self, data_item, model_output):
        """执行AIME案例评估
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
            evaluator_name='AIMEEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
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

