from .base import BaseBackend
import asyncio

class AgentBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'agent'
        self.eval_type = config.get('eval_type', 'model')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def execute(self, model, case, response=None):
        """执行Agent案例评估
        输入：案例dict，包含prompt、answer、metadata字段，metadata中可包含tools信息
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
        """异步执行Agent案例评估"""
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
        """基于规则的Agent评测"""
        if not response:
            # 获取工具信息
            tools = case.get('metadata', {}).get('tools', [])
            # 生成Agent响应
            response = model.generate_with_tools(case['prompt'], tools)
        
        # 简单的评分逻辑：检查参考答案是否在响应中
        if case['answer'] in response:
            return 1.0
        else:
            return 0.0
    
    def _model_based_evaluation(self, model, case, response=None):
        """基于大模型的Agent评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not response:
            # 获取工具信息
            tools = case.get('metadata', {}).get('tools', [])
            # 生成Agent响应
            response = model.generate_with_tools(case['prompt'], tools)
        
        # 使用Agent专用的评判提示词模板
        judge_prompt = (f"请评估以下Agent对任务的完成情况。\n\n" 
                       f"任务描述：{case['prompt']}\n\n" 
                       f"可用工具：{case.get('metadata', {}).get('tools', [])}\n\n" 
                       f"Agent执行过程：{response}\n\n" 
                       f"参考答案：{case['answer']}\n\n" 
                       f"评估标准：\n" 
                       f"1. 工具调用是否正确（选择了合适的工具，参数正确）\n" 
                       f"2. 多轮交互是否有效（信息收集充分，决策合理）\n" 
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
        """基于大模型的异步Agent评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not response:
            # 获取工具信息
            tools = case.get('metadata', {}).get('tools', [])
            # 生成Agent响应
            response = await model.async_generate_with_tools(case['prompt'], tools)
        
        # 使用Agent专用的评判提示词模板
        judge_prompt = (f"请评估以下Agent对任务的完成情况。\n\n" 
                       f"任务描述：{case['prompt']}\n\n" 
                       f"可用工具：{case.get('metadata', {}).get('tools', [])}\n\n" 
                       f"Agent执行过程：{response}\n\n" 
                       f"参考答案：{case['answer']}\n\n" 
                       f"评估标准：\n" 
                       f"1. 工具调用是否正确（选择了合适的工具，参数正确）\n" 
                       f"2. 多轮交互是否有效（信息收集充分，决策合理）\n" 
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
            'backend_type': 'agent',
            'task_type': self.task_type
        }
