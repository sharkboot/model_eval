from .base import BaseEvaluator
from core.base import EvaluationResult

class WritingBenchEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.task_type = 'writing'
        self.eval_type = config.get('eval_type', 'model')  # rule 或 model
        self.judge_model = config.get('judge_model', None)  # 裁判模型
    
    def evaluate(self, data_item, model_output):
        """执行WritingBench案例评估
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
            evaluator_name='WritingBenchEvaluator',
            raw_output=model_output,
            metrics=metrics,
            details={'task_type': self.task_type, 'eval_type': self.eval_type}
        )
    
    def _rule_based_evaluation(self, data_item, model_output):
        """基于规则的WritingBench评测"""
        if not model_output:
            return 0.0
        
        # 简单的评分逻辑：根据长度和关键词匹配
        score = 0.0
        
        # 检查长度要求
        length_requirement = data_item.metadata.get('length', '')
        if length_requirement:
            # 简单的长度检查
            if len(model_output) > 500:
                score += 0.3
        
        # 检查是否包含关键词
        prompt = data_item.prompt
        if any(keyword in model_output for keyword in prompt.split()[:10]):
            score += 0.3
        
        # 检查响应是否为空
        if model_output and len(model_output) > 100:
            score += 0.4
        
        return min(score, 1.0)
    
    def _model_based_evaluation(self, data_item, model_output):
        """基于大模型的WritingBench评测（使用裁判模型）"""
        if not self.judge_model:
            raise ValueError("Judge model is required for model-based evaluation")
        
        if not model_output:
            return 0.0
        
        # 获取评估标准
        criteria = data_item.metadata.get('criteria', [])
        if not criteria:
            # 如果没有评估标准，生成默认的评估标准
            criteria = self._generate_default_criteria(data_item.prompt)
        
        # 使用WritingBench专用的评判提示词模板
        judge_prompt = self._construct_judge_prompt(data_item.prompt, model_output, criteria)
        
        # 调用裁判模型
        judge_response = self.judge_model.generate([{'prompt': judge_prompt}])[0]
        
        # 解析裁判模型的回答
        try:
            judge_answer = str(judge_response)
            
            # 提取评分
            import re
            score_match = re.search(r'\b([0-9]+(\.[0-9]+)?)\b', judge_answer)
            if score_match:
                return float(score_match.group(1)) / 10.0  # WritingBench使用10分制，转换为0-1.0
            else:
                return 0.0
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return 0.0
    
    def _generate_default_criteria(self, prompt):
        """生成默认的评估标准"""
        return [
            "内容相关性：写作内容是否与任务要求相关",
            "逻辑连贯性：写作内容是否逻辑清晰，结构合理",
            "语言表达：语言是否流畅，用词是否准确",
            "创意性：是否有独特的观点或创意",
            "格式符合要求：是否符合指定的格式和长度要求"
        ]
    
    def _construct_judge_prompt(self, prompt, response, criteria):
        """构建WritingBench的评判提示词"""
        criteria_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])
        
        judge_prompt = (f"请作为专业的写作评委，根据以下评估标准对给定的写作内容进行评分。\n\n" 
                       f"写作任务：{prompt}\n\n" 
                       f"评估标准：\n{criteria_text}\n\n" 
                       f"写作内容：{response}\n\n" 
                       f"评分要求：\n" 
                       f"1. 对每个评估标准进行评分（0-10分）\n" 
                       f"2. 计算总分并给出最终评分（0-10分）\n" 
                       f"3. 提供简要的评分理由\n\n" 
                       f"请按照以下格式输出：\n" 
                       f"最终评分：[分数]\n" 
                       f"评分理由：[理由]\n")
        
        return judge_prompt

