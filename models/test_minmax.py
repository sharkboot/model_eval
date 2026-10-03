from core.base import ModelOutput, ModelInput
from core.registry import Registry
from models.base import BaseModel


@Registry.register("MiniMax", "model")
class LocalModel(BaseModel):
    def generate(self, model_input: ModelInput) -> ModelOutput:
        from openai import OpenAI
        import os
        client = OpenAI(
            api_key=self.config.get("api_key") or os.environ.get("MINIMAX_API_KEY", ""),
            base_url=self.config.get("base_url", "https://api.ttxxvv.cn/v1"),
        )
        resp = client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=[
                {"role": "user", "content": model_input.prompt}
            ],
            **self.config.get("generation_config")
        )
        return ModelOutput(type="text", text=resp.choices[0].message.content)
