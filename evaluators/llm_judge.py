from abc import ABC, abstractmethod
from core.base import DataItem, EvaluationResult
from core.registry import Registry
from evaluators.base import BaseEvaluator

SYSTEM_PROMPT = """
你是严谨问答判定专家，仅根据问题与标准答案，判断用户回答是否正确。
完全贴合核心要点 = Yes，答非所问、遗漏关键内容、逻辑错误 = No。
问题：{question}
标准答案：{standard_answer}
用户回答：{user_reply}
严格只输出：Yes 或 No，不添加任何多余文字、理由、解释。
"""


@Registry.register("llm_judge", "evaluator")
class LLMJudgeEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.config = config

    def evaluate(self, pred: str, item: DataItem):
        from openai import OpenAI
        client = OpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url")  # 兼容OpenAI格式
        )
        prompt = SYSTEM_PROMPT.replace("question", item.prompt).replace("standard_answer", item.reference).replace(
            "user_reply", pred)
        resp = client.chat.completions.create(
            model=self.config.get("model"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            **self.config.get("generation_config", {})
        )

        return {"llm_judge": 1.0 if "Yes" in resp.choices[0].message.content else 0}
