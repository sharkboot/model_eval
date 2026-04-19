from core.base import ModelInput
from core.registry import Registry


class BasePromptBuilder:
    def __init__(self, config):
        self.config = config

@Registry.register("qa_builder", "prompt_builder")
class QAPromptBuilder(BasePromptBuilder):

    def build(self, item):
        return ModelInput(
            type="text",
            prompt=f"请回答以下问题：\n{item.prompt}"
        )
