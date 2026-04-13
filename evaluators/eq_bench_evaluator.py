from .base import BaseEvaluator
from core.base import EvaluationResult
import math

class EQBenchEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'emotional_intelligence'
        self.eval_type = config.get('eval_type', 'rule')  # 对于EQ-Bench，主要使用规则评测
    
    def evaluate(self, data_item, model_output):
        """执行EQ-Bench案例评估
        输入：DataItem和模型输出
        输出：EvaluationResult
        """
        try:
            if not model_output:
                score = 0.0
            else:
                # 提取模型预测的评分
                emotions = data_item.metadata.get('emotions', [])
                predicted_ratings = self._extract_ratings(model_output, emotions)
                
                # 与参考评分比较，计算相似度
                reference_ratings = data_item.reference
                score = self._calculate_similarity(predicted_ratings, reference_ratings)
        except Exception as e:
            score = 0.0
        
        metrics = {'score': score, 'accuracy': score}
        
        return EvaluationResult(
            data_id=data_item.id,
            evaluator_name='EQBenchEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
    def _extract_ratings(self, response, emotions):
        """从模型响应中提取情绪评分"""
        import re
        ratings = {}
        
        for emotion in emotions:
            # 匹配情绪: 分数的格式
            pattern = f"{emotion}[:：]\s*(\d+)"
            match = re.search(pattern, response)
            if match:
                try:
                    ratings[emotion] = int(match.group(1))
                except ValueError:
                    ratings[emotion] = 0
            else:
                ratings[emotion] = 0
        
        return ratings
    
    def _calculate_similarity(self, predicted, reference):
        """计算预测评分与参考评分的相似度
        使用EQ-Bench的评分算法
        """
        total_error = 0
        num_emotions = len(reference)
        
        for emotion, ref_score in reference.items():
            pred_score = predicted.get(emotion, 0)
            # 计算绝对误差
            error = abs(pred_score - ref_score)
            total_error += error
        
        # 计算平均误差
        avg_error = total_error / num_emotions if num_emotions > 0 else 0
        
        # 转换为相似度分数（0-1之间）
        # 误差越小，分数越高
        similarity = max(0, 1 - (avg_error / 10))
        
        return similarity

