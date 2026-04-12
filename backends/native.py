from .base import BaseEvaluator
import time
import asyncio
from utils.data_classes import EvaluationResult

class NativeEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'llm')
        self.eval_type = config.get('eval_type', 'rule')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, data_item, model_output):
        """执行案例评估
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
        
        metrics = {'score': score}
        if self.task_type == 'llm':
            metrics['accuracy'] = score
        
        return EvaluationResult(
            data_id=data_item.id,
            evaluator_name='NativeEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
    def _rule_based_evaluation(self, data_item, model_output):
        """基于规则的评测"""
        if self.task_type == 'llm':
            # 简单的评分逻辑：检查参考答案是否在响应中
            if str(data_item.reference) in model_output:
                return 1.0
            else:
                return 0.0
        elif self.task_type == 'vlm':
            # VLM评估逻辑
            if str(data_item.reference) in model_output:
                return 1.0
            else:
                return 0.0
        elif self.task_type == 'embedding':
            # Embedding评估逻辑（示例）
            return 0.5  # 示例分数
        elif self.task_type == 'reranker':
            # Reranker评估逻辑（示例）
            return 0.7  # 示例分数
        elif self.task_type == 'aigc':
            # AIGC评估逻辑（示例）
            return 0.8  # 示例分数
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _model_based_evaluation(self, data_item, model_output):
        """基于大模型的评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        # 使用通用的评判提示词模板
        judge_prompt = f"请评估以下模型对问题的回答是否正确。\n\n问题：{data_item.prompt}\n\n参考答案：{data_item.reference}\n\n模型回答：{model_output}\n\n评估标准：如果模型回答与参考答案意思一致，返回1.0；否则返回0.0。"
        judge_response = self.judge_model.generate([{'prompt': judge_prompt}])[0]
        
        # 解析裁判模型的回答
        try:
            judge_answer = str(judge_response)
            
            # 调试信息
            print(f"Judge response: {judge_answer}")
            
            if '1.0' in judge_answer or '正确' in judge_answer:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return 0.0

