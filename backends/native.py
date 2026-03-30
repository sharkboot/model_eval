from .base import BaseBackend
import time

class NativeBackend(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = config.get('task_type', 'llm')
        self.eval_type = config.get('eval_type', 'rule')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, model, dataset, max_samples=None):
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
                       f"问题: {case['prompt']}\n" 
                       f"正确答案: {case['answer']}\n" 
                       f"预测答案: {response}\n\n" 
                       f"将此新问题的预测答案评定为以下之一：\n" 
                       f"A:【正确】\n" 
                       f"B:【错误】\n" 
                       f"C:【未尝试】\n\n" 
                       f"只返回字母\"A\"、\"B\"或\"C\"，无须添加其他文本。")
        
        # 调用裁判模型
        judge_response = self.judge_model.generate(judge_prompt)
        
        # 解析裁判模型的回答
        try:
            if isinstance(judge_response, dict) and 'choices' in judge_response:
                judge_answer = judge_response['choices'][0]['text'].strip()
            else:
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
    
    def get_backend_info(self):
        return {
            'backend_type': 'native',
            'task_type': self.task_type
        }
