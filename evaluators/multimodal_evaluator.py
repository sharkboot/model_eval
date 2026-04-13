from .base import BaseEvaluator
import asyncio
from core.base import EvaluationResult

class MultimodalEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'vlm'  # Visual Language Model
        self.eval_type = config.get('eval_type', 'model')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, data_item, model_output):
        """执行多模态案例评估
        输入：DataItem和模型输出
        输出：EvaluationResult
        """
        try:
            if self.eval_type == 'rule':
                # 规则评测
                score = self._rule_based_evaluation(data_item, model_output)
            elif self.eval_type == 'model':
                # 大模型评测（使用裁判模型）
                score = self._model_based_evaluation(data_item, model_output)
            else:
                raise ValueError(f"Unsupported evaluation type: {self.eval_type}")
        except Exception as e:
            score = 0.0
        
        metrics = {'score': score, 'accuracy': score}
        
        return EvaluationResult(
            data_id=data_item.id,
            evaluator_name='MultimodalEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
    def _rule_based_evaluation(self, data_item, model_output):
        """基于规则的多模态评测"""
        # 简单的评分逻辑：检查参考答案是否在响应中
        if str(data_item.reference) in model_output:
            return 1.0
        else:
            return 0.0
    
    def _model_based_evaluation(self, data_item, model_output):
        """基于大模型的多模态评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        # 使用多模态专用的评判提示词模板
        judge_prompt = (f"请评估以下多模态模型对任务的完成情况。\n\n" 
                       f"任务描述：{data_item.prompt}\n\n" 
                       f"多模态输入：{data_item.metadata.get('image_url', '无')}\n\n" 
                       f"模型回答：{model_output}\n\n" 
                       f"参考答案：{data_item.reference}\n\n" 
                       f"评估标准：\n" 
                       f"1. 对多模态输入的理解是否正确\n" 
                       f"2. 回答是否符合任务要求\n" 
                       f"3. 最终结果是否正确（与参考答案一致）\n\n" 
                       f"请综合以上因素，给出0-1.0的评分，其中1.0表示完全正确。")
        
        # 调用裁判模型
        judge_response = self.judge_model.generate([{'prompt': judge_prompt}])[0]
        
        # 解析裁判模型的回答
        try:
            judge_answer = str(judge_response)
            
            # 提取评分
            import re
            score_match = re.search(r'\b([0-9]+(\.[0-9]+)?)\b', judge_answer)
            if score_match:
                return float(score_match.group(1))
            else:
                return 0.0
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return 0.0

