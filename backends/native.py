from .base import BaseBackend
import time
import asyncio

class NativeBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'llm')
        self.eval_type = config.get('eval_type', 'rule')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, model, dataset, max_samples=None, num_runs=1, scoring_strategy='highest'):
        results = []
        dataset.load()
        data = dataset.get_data()
        
        # 限制样本数量
        if max_samples is not None:
            data = data[:max_samples]
        
        for i, item in enumerate(data):
            start_time = time.time()
            try:
                # 转换为标准案例格式
                case = dataset.convert_to_case(item)
                
                # 执行多次评测
                runs = []
                for run in range(num_runs):
                    # 生成模型响应
                    response = model.generate(case['prompt'])
                    # 执行评估
                    score = self.execute(model, case, response)
                    runs.append({
                        'run_id': run,
                        'output': response,
                        'score': score
                    })
                
                # 根据策略确定最终得分
                final_score = self._determine_final_score(runs, scoring_strategy)
                
                end_time = time.time()
                results.append({
                    'id': i,
                    'input': item,
                    'case': case,
                    'runs': runs,  # 所有运行的结果
                    'final_score': final_score,  # 最终得分
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
    
    async def async_evaluate(self, model, dataset, max_samples=None, num_runs=1, scoring_strategy='highest', concurrency_limit=5):
        """异步评测方法"""
        results = []
        dataset.load()
        data = dataset.get_data()
        
        # 限制样本数量
        if max_samples is not None:
            data = data[:max_samples]
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        # 定义处理单个样本的异步函数
        async def process_item(i, item):
            async with semaphore:
                start_time = time.time()
                try:
                    # 转换为标准案例格式
                    case = dataset.convert_to_case(item)
                    
                    # 执行多次评测
                    runs = []
                    # 并发执行多次评测
                    run_tasks = []
                    for run in range(num_runs):
                        run_tasks.append(self._async_run_evaluation(model, case, run))
                    runs = await asyncio.gather(*run_tasks)
                    
                    # 根据策略确定最终得分
                    final_score = self._determine_final_score(runs, scoring_strategy)
                    
                    end_time = time.time()
                    return {
                        'id': i,
                        'input': item,
                        'case': case,
                        'runs': runs,  # 所有运行的结果
                        'final_score': final_score,  # 最终得分
                        'latency': end_time - start_time
                    }
                except Exception as e:
                    end_time = time.time()
                    return {
                        'id': i,
                        'input': item,
                        'output': None,
                        'error': str(e),
                        'latency': end_time - start_time
                    }
        
        # 并发处理所有样本
        tasks = []
        for i, item in enumerate(data):
            tasks.append(process_item(i, item))
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _async_run_evaluation(self, model, case, run_id):
        """异步执行单次评测"""
        # 生成模型响应
        response = await model.async_generate(case['prompt'])
        # 执行评估
        score = await self.async_execute(model, case, response)
        return {
            'run_id': run_id,
            'output': response,
            'score': score
        }
    
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
        """将评分转换为数值"""
        if score == 'A':
            return 2
        elif score == 'B':
            return 1
        elif score == 'C':
            return 0
        else:
            return 0
    
    def _value_to_score(self, value):
        """将数值转换为评分"""
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
