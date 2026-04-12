from .base import BaseEvaluator
import asyncio
from utils.data_classes import EvaluationResult

class ChineseSimpleQAEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'llm'
        self.eval_type = 'model'
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, data_item, model_output):
        """执行案例评估
        输入：DataItem和模型输出
        输出：EvaluationResult
        """
        try:
            if self.eval_type == 'model':
                # 大模型评测（使用裁判模型）
                score = self._model_based_evaluation(data_item, model_output)
            else:
                # 其他评测类型
                score = 'C'  # 默认未尝试
        except Exception as e:
            score = 'C'  # 出错时返回未尝试
        
        # 计算指标
        metrics = {'score': score}
        if score == 'A':
            metrics['correct'] = 1
            metrics['incorrect'] = 0
            metrics['not_attempted'] = 0
        elif score == 'B':
            metrics['correct'] = 0
            metrics['incorrect'] = 1
            metrics['not_attempted'] = 0
        else:  # 'C'
            metrics['correct'] = 0
            metrics['incorrect'] = 0
            metrics['not_attempted'] = 1
        
        return EvaluationResult(
            data_id=data_item.id,
            evaluator_name='ChineseSimpleQAEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
    def _model_based_evaluation(self, data_item, model_output):
        """基于大模型的评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        # 使用Chinese SimpleQA的评判提示词模板
        judge_prompt = (f"请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。\n" 
                       f"您的任务是将结果评定为：【正确】、【错误】或【未尝试】。\n\n" 
                       f"以下是【正确】的答复示例：\n" 
                       f"问题：贝拉克·奥巴马的孩子叫什么名字？\n" 
                       f"标准答案：玛丽亚·奥巴马和萨莎·奥巴马\n" 
                       f"模型预测1：Malia Obama and Sasha Obama\n" 
                       f"（完整包含参考答案且不矛盾）→ 评定为【正确】\n\n" 
                       f"以下是【错误】的答复示例：\n" 
                       f"问题：巴拉克·奥巴马的孩子叫什么名字？\n" 
                       f"标准答案：玛丽亚·奥巴马和萨莎·奥巴马\n" 
                       f"模型预测：玛丽亚、萨莎和苏珊\n" 
                       f"（包含矛盾信息）→ 评定为【错误】\n\n" 
                       f"以下是【未尝试】的答复示例：\n" 
                       f"问题：巴拉克·奥巴马的孩子叫什么名字？\n" 
                       f"标准答案：玛丽亚·奥巴马和萨莎·奥巴马\n" 
                       f"模型预测：我不知道。\n" 
                       f"（未包含参考答案但也不矛盾）→ 评定为【未尝试】\n\n" 
                       f"问题: {data_item.prompt}\n" 
                       f"正确答案: {data_item.reference}\n" 
                       f"预测答案: {model_output}\n\n" 
                       f"将此新问题的预测答案评定为以下之一：\n" 
                       f"A:【正确】\n" 
                       f"B:【错误】\n" 
                       f"C:【未尝试】\n\n" 
                       f"只返回字母\"A\"、\"B\"或\"C\"，无须添加其他文本。")
        
        # 调用裁判模型
        judge_response = self.judge_model.generate([{'prompt': judge_prompt}])[0]
        
        # 解析裁判模型的回答
        try:
            judge_answer = str(judge_response)
            
            # 调试信息
            print(f"Judge response: {judge_answer}")
            
            # 返回对应的评分
            if 'A' in judge_answer:
                return 'A'  # 正确
            elif 'B' in judge_answer:
                return 'B'  # 错误
            elif 'C' in judge_answer:
                return 'C'  # 未尝试
            else:
                return 'C'  # 默认未尝试
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return 'C'  # 出错时返回未尝试

