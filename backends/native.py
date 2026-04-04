from .base import BaseBackend
import time
import asyncio

class NativeBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'llm')
        self.eval_type = config.get('eval_type', 'rule')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    async def async_execute(self, model, case, response=None):
        """异步执行案例评估"""
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
    
    async def _async_model_based_evaluation(self, model, case, response=None):
        """基于大模型的异步评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not response:
            response = await model.async_generate(case['prompt'])
        
        # 使用通用的评判提示词模板
        judge_prompt = f"请评估以下模型对问题的回答是否正确。\n\n问题：{case['prompt']}\n\n参考答案：{case['answer']}\n\n模型回答：{response}\n\n评估标准：如果模型回答与参考答案意思一致，返回1.0；否则返回0.0。"
        judge_response = await self.judge_model.async_generate(judge_prompt)
        
        # 解析裁判模型的回答
        try:
            if isinstance(judge_response, dict) and 'choices' in judge_response:
                judge_answer = judge_response['choices'][0]['text'].strip()
            else:
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
    
    def _determine_final_score(self, runs, strategy):
        """根据策略确定最终得分"""
        scores = [run['score'] for run in runs]
        
        if strategy == 'highest':
            # 取最高分
            return max(scores, key=self._score_to_value)
        elif strategy == 'lowest':
            # 取最低分
            return min(scores, key=self._score_to_value)
        elif strategy == 'average':
            # 取平均值
            score_values = [self._score_to_value(score) for score in scores]
            avg_value = sum(score_values) / len(score_values)
            return self._value_to_score(avg_value)
        elif strategy == 'median':
            # 取中位数
            score_values = [self._score_to_value(score) for score in scores]
            score_values.sort()
            mid = len(score_values) // 2
            median_value = score_values[mid] if len(score_values) % 2 == 1 else (score_values[mid-1] + score_values[mid]) / 2
            return self._value_to_score(median_value)
        else:
            # 默认取最高分
            return max(scores, key=self._score_to_value)
    
    def _score_to_value(self, score):
        """将评分转换为数值，支持多种类型"""
        # 处理字符串类型
        if isinstance(score, str):
            if score == 'A':
                return 2
            elif score == 'B':
                return 1
            elif score == 'C':
                return 0
            else:
                return 0
        # 处理数值类型
        elif isinstance(score, (int, float)):
            return float(score)
        # 处理字典类型
        elif isinstance(score, dict):
            return score.get('value', 0)
        # 其他类型
        else:
            return 0
    
    def _value_to_score(self, value):
        """将数值转换为评分，支持多种类型"""
        # 如果原始评分是字符串类型，转换为字符串
        if value >= 1.5:
            return 'A'
        elif value >= 0.5:
            return 'B'
        else:
            return 'C'
    
    def execute(self, model, case, response=None):
        """执行案例评估
        输入：案例dict，包含prompt、answer、metadata字段
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
    
    def _rule_based_evaluation(self, model, case, response=None):
        """基于规则的评测"""
        if not response:
            response = model.generate(case['prompt'])
        
        if self.task_type == 'llm':
            # 简单的评分逻辑：检查参考答案是否在响应中
            if case['answer'] in response:
                return 1.0
            else:
                return 0.0
        elif self.task_type == 'vlm':
            # VLM评估逻辑
            if case['answer'] in response:
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
    
    def _model_based_evaluation(self, model, case, response=None):
        """基于大模型的评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not response:
            response = model.generate(case['prompt'])
        
        # 使用通用的评判提示词模板
        judge_prompt = f"请评估以下模型对问题的回答是否正确。\n\n问题：{case['prompt']}\n\n参考答案：{case['answer']}\n\n模型回答：{response}\n\n评估标准：如果模型回答与参考答案意思一致，返回1.0；否则返回0.0。"
        judge_response = self.judge_model.generate(judge_prompt)
        
        # 解析裁判模型的回答
        try:
            if isinstance(judge_response, dict) and 'choices' in judge_response:
                judge_answer = judge_response['choices'][0]['text'].strip()
            else:
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
    
    def get_backend_info(self):
        return {
            'backend_type': 'native',
            'task_type': self.task_type
        }
