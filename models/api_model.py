from openai import OpenAI
import json

class APIModel:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model_name = config.get('model_name')
    
    def generate(self, prompt):
        raise NotImplementedError("Subclass must implement generate method")
    
    async def async_generate(self, prompt):
        raise NotImplementedError("Subclass must implement async_generate method")
    
    def generate_with_tools(self, prompt, tools):
        """使用工具调用生成响应"""
        # 默认实现：不使用工具，直接生成
        return self.generate(prompt)
    
    async def async_generate_with_tools(self, prompt, tools):
        """异步使用工具调用生成响应"""
        # 默认实现：不使用工具，直接生成
        return await self.async_generate(prompt)
    
    def generate_multimodal(self, prompt, image_url):
        """生成多模态响应"""
        # 默认实现：忽略图像，直接生成
        return self.generate(prompt)
    
    async def async_generate_multimodal(self, prompt, image_url):
        """异步生成多模态响应"""
        # 默认实现：忽略图像，直接生成
        return await self.async_generate(prompt)
    
    def get_model_info(self):
        return {
            'model_type': 'api',
            'model_name': self.model_name,
            'base_url': self.base_url
        }

class OpenAIModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    async def async_generate(self, prompt):
        try:
            # 由于OpenAI客户端的async方法可能不可用，这里使用同步方法
            # 在实际应用中，可以使用aiohttp等库实现真正的异步调用
            return self.generate(prompt)
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    def generate_with_tools(self, prompt, tools):
        """使用工具调用生成响应"""
        try:
            # 构建包含工具信息的提示词
            tool_info = json.dumps(tools, ensure_ascii=False)
            tool_prompt = f"你可以使用以下工具来完成任务：\n{tool_info}\n\n任务：{prompt}\n\n请根据需要使用工具来完成任务，并提供最终答案。"
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': tool_prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    async def async_generate_with_tools(self, prompt, tools):
        """异步使用工具调用生成响应"""
        try:
            # 构建包含工具信息的提示词
            tool_info = json.dumps(tools, ensure_ascii=False)
            tool_prompt = f"你可以使用以下工具来完成任务：\n{tool_info}\n\n任务：{prompt}\n\n请根据需要使用工具来完成任务，并提供最终答案。"
            
            # 由于OpenAI客户端的async方法可能不可用，这里使用同步方法
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': tool_prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    def generate_multimodal(self, prompt, image_url):
        """生成多模态响应"""
        try:
            # 构建包含图像信息的提示词
            multimodal_prompt = f"图片URL: {image_url}\n\n{prompt}"
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': multimodal_prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    async def async_generate_multimodal(self, prompt, image_url):
        """异步生成多模态响应"""
        try:
            # 构建包含图像信息的提示词
            multimodal_prompt = f"图片URL: {image_url}\n\n{prompt}"
            
            # 由于OpenAI客户端的async方法可能不可用，这里使用同步方法
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': multimodal_prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"

class ClaudeModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以初始化Claude的客户端
    
    def generate(self, prompt):
        # Claude的实现
        return "Claude response: " + prompt
    
    async def async_generate(self, prompt):
        # Claude的异步实现
        return "Claude async response: " + prompt

class GenericAPIModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以初始化通用API客户端
    
    def generate(self, prompt):
        # 通用API的实现
        return "Generic API response: " + prompt
    
    async def async_generate(self, prompt):
        # 通用API的异步实现
        return "Generic API async response: " + prompt
