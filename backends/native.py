from .base import BaseBackend
import time

class NativeBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'llm')
        self.eval_type = config.get('eval_type', 'rule')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, model, dataset):
        results = []
        dataset.load()
        data = dataset.get_data()
        
        for i, item in enumerate(data):
            start_time = time.time()
            try:
                # 转换为标准案例格式
                case = dataset.convert_to_case(item)
                # 生成模型响应
                response = model.generate(case['prompt'])
                # 执行评估
                score = self.execute(model, case, response)
                
                end_time = time.time()
                results.append({
                    'id': i,
                    'input': item,
                    'case': case,
                    'output': response,  # 使用模型的实际响应作为输出
                    'score': score,
                    'latency': end_time - start_time
                })
            except Exception as e:
                end_time = time.time()
                results.append({
                    'id': i,
                    'input': item,
                    'output': None,
                    'error': str(e),
                    'latency': end_time - start_time
                })
        
        return results
    
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
        
        # 使用裁判模型进行评估
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
