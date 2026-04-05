from .base import BaseBackend
import asyncio

class MultimodalBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'vlm'  # Visual Language Model
        self.eval_type = config.get('eval_type', 'model')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def execute(self, model, case, response=None):
        """执行多模态案例评估
        输入：案例dict，包含prompt、answer、metadata字段，metadata中可包含image_url等多模态信息
        输出：评估分数
        """
        try:
            if self.eval_type == 'rule':
                # 规则评测
                return self._rule_based_evaluation(model, case, response)
            elif self.eval_type == 'model':
                # 大模型评测（使用裁判模型）
                return self._model_based_evaluation(model, case, response)
            else:
                raise ValueError(f"Unsupported evaluation type: {self.eval_type}")
        except Exception as e:
            return 0.0
    
    async def async_execute(self, model, case, response=None):
        """异步执行多模态案例评估"""
        try:
            if self.eval_type == 'rule':
                # 规则评测
                return self._rule_based_evaluation(model, case, response)
            elif self.eval_type == 'model':
                # 大模型评测（使用裁判模型）
                return await self._async_model_based_evaluation(model, case, response)
            else:
                raise ValueError(f"Unsupported evaluation type: {self.eval_type}")
        except Exception as e:
            return 0.0
    
    def _rule_based_evaluation(self, model, case, response=None):
        """基于规则的多模态评测"""
        if not response:
            # 获取多模态信息
            image_url = case.get('metadata', {}).get('image_url', '')
            # 生成多模态模型响应
            response = model.generate_multimodal(case['prompt'], image_url)
        
        # 简单的评分逻辑：检查参考答案是否在响应中
        if case['answer'] in response:
            return 1.0
        else:
            return 0.0
    
    def _model_based_evaluation(self, model, case, response=None):
        """基于大模型的多模态评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not response:
            # 获取多模态信息
            image_url = case.get('metadata', {}).get('image_url', '')
            # 生成多模态模型响应
            response = model.generate_multimodal(case['prompt'], image_url)
        
        # 使用多模态专用的评判提示词模板
        judge_prompt = (f"请评估以下多模态模型对任务的完成情况。\n\n" 
                       f"任务描述：{case['prompt']}\n\n" 
                       f"多模态输入：{case.get('metadata', {}).get('image_url', '无')}\n\n" 
                       f"模型回答：{response}\n\n" 
                       f"参考答案：{case['answer']}\n\n" 
                       f"评估标准：\n" 
                       f"1. 对多模态输入的理解是否正确\n" 
                       f"2. 回答是否符合任务要求\n" 
                       f"3. 最终结果是否正确（与参考答案一致）\n\n" 
                       f"请综合以上因素，给出0-1.0的评分，其中1.0表示完全正确。")
        
        # 调用裁判模型
        judge_response = self.judge_model.generate(judge_prompt)
        
        # 解析裁判模型的回答
        try:
            if isinstance(judge_response, dict) and 'choices' in judge_response:
                judge_answer = judge_response['choices'][0]['text'].strip()
            else:
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
    
    async def _async_model_based_evaluation(self, model, case, response=None):
        """基于大模型的异步多模态评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not response:
            # 获取多模态信息
            image_url = case.get('metadata', {}).get('image_url', '')
            # 生成多模态模型响应
            response = await model.async_generate_multimodal(case['prompt'], image_url)
        
        # 使用多模态专用的评判提示词模板
        judge_prompt = (f"请评估以下多模态模型对任务的完成情况。\n\n" 
                       f"任务描述：{case['prompt']}\n\n" 
                       f"多模态输入：{case.get('metadata', {}).get('image_url', '无')}\n\n" 
                       f"模型回答：{response}\n\n" 
                       f"参考答案：{case['answer']}\n\n" 
                       f"评估标准：\n" 
                       f"1. 对多模态输入的理解是否正确\n" 
                       f"2. 回答是否符合任务要求\n" 
                       f"3. 最终结果是否正确（与参考答案一致）\n\n" 
                       f"请综合以上因素，给出0-1.0的评分，其中1.0表示完全正确。")
        
        # 调用裁判模型
        judge_response = await self.judge_model.async_generate(judge_prompt)
        
        # 解析裁判模型的回答
        try:
            if isinstance(judge_response, dict) and 'choices' in judge_response:
                judge_answer = judge_response['choices'][0]['text'].strip()
            else:
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
    
    def get_backend_info(self):
        return {
            'backend_type': 'multimodal',
            'task_type': self.task_type
        }
