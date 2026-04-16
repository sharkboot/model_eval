from openai import OpenAI
import json
from .base import BaseModel
from core.base import ModelInput
from .registry import ModelRegistry

@ModelRegistry.register('APIModel')
class APIModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model_name = config.get('model_name')
    
    def generate(self, inputs):
        raise NotImplementedError("Subclass must implement generate method")
    
    async def async_generate(self, prompt):
        raise NotImplementedError("Subclass must implement async_generate method")
    
    def generate_with_tools(self, prompt, tools):
        """使用工具调用生成响应"""
        # 默认实现：不使用工具，直接生成
        return self.generate([ModelInput(prompt=prompt)])[0]
    
    async def async_generate_with_tools(self, prompt, tools):
        """异步使用工具调用生成响应"""
        # 默认实现：不使用工具，直接生成
        return await self.async_generate(prompt)
    
    def generate_multimodal(self, prompt, image_url):
        """生成多模态响应"""
        # 默认实现：忽略图像，直接生成
        return self.generate([ModelInput(prompt=prompt)])[0]
    
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

@ModelRegistry.register('OpenAIModel')
class OpenAIModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(self, inputs):
        results = []
        for input_item in inputs:
            try:
                messages = []
                if input_item.system_prompt:
                    messages.append({'role': 'system', 'content': input_item.system_prompt})
                
                # 如果有对话历史，使用对话历史
                if input_item.messages:
                    messages.extend(input_item.messages)
                # 添加当前提示作为最新的用户消息
                messages.append({'role': 'user', 'content': input_item.prompt})
                
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **input_item.generation_config
                )
                results.append(completion.choices[0].message.content)
            except Exception as e:
                print(f"API call failed: {e}")
                results.append("我不知道。")
        return results
    
    async def async_generate(self, prompt):
        try:
            # 由于OpenAI客户端的async方法可能不可用，这里使用同步方法
            # 在实际应用中，可以使用aiohttp等库实现真正的异步调用
            return self.generate([ModelInput(prompt=prompt)])[0]
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    def generate_with_tools(self, prompt, tools):
        """使用工具调用生成响应"""
        try:
            # 构建包含工具信息的提示词
            tool_info = json.dumps(tools, ensure_ascii=False)
            tool_prompt = f"你可以使用以下工具来完成任务：\n{tool_info}\n\n任务：{prompt}\n\n请根据需要使用工具来完成任务，并提供最终答案。"
            
            return self.generate([ModelInput(prompt=tool_prompt)])[0]
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
            return self.generate([ModelInput(prompt=tool_prompt)])[0]
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    def generate_multimodal(self, prompt, image_url):
        """生成多模态响应"""
        try:
            # 构建包含图像信息的提示词
            multimodal_prompt = f"图片URL: {image_url}\n\n{prompt}"
            
            return self.generate([ModelInput(prompt=multimodal_prompt)])[0]
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"
    
    async def async_generate_multimodal(self, prompt, image_url):
        """异步生成多模态响应"""
        try:
            # 构建包含图像信息的提示词
            multimodal_prompt = f"图片URL: {image_url}\n\n{prompt}"
            
            # 由于OpenAI客户端的async方法可能不可用，这里使用同步方法
            return self.generate([ModelInput(prompt=multimodal_prompt)])[0]
        except Exception as e:
            print(f"API call failed: {e}")
            return "我不知道。"

@ModelRegistry.register('ClaudeModel')
class ClaudeModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以初始化Claude的客户端
    
    def generate(self, inputs):
        results = []
        for input_item in inputs:
            # Claude的实现，支持多轮对话
            if input_item.messages:
                # 如果有对话历史，构建完整的对话上下文
                conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in input_item.messages])
                results.append(f"Claude response to conversation: {conversation}\nUser: {input_item.prompt}")
            else:
                results.append("Claude response: " + input_item.prompt)
        return results
    
    async def async_generate(self, prompt):
        # Claude的异步实现
        return "Claude async response: " + prompt

@ModelRegistry.register('GenericAPIModel')
class GenericAPIModel(APIModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以初始化通用API客户端
    
    def generate(self, inputs):
        results = []
        for input_item in inputs:
            # 通用API的实现，支持多轮对话
            if input_item.messages:
                # 如果有对话历史，构建完整的对话上下文
                conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in input_item.messages])
                results.append(f"Generic API response to conversation: {conversation}\nUser: {input_item.prompt}")
            else:
                results.append("Generic API response: " + input_item.prompt)
        return results
    
    async def async_generate(self, prompt):
        # 通用API的异步实现
        return "Generic API async response: " + prompt
