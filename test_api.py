import os
import httpx
from openai import OpenAI

# 注意: 不同地域的base_url不通用（下方示例使用北京地域的 base_url）
# - 华北2（北京）: https://dashscope.aliyuncs.com/compatible-mode/v1
# - 美国（弗吉尼亚）: https://dashscope-us.aliyuncs.com/compatible-mode/v1
# - 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# 创建一个带有更多调试信息的HTTP客户端
http_client = httpx.Client(
    verify=False,  # 禁用SSL验证，可能解决SSL问题
    timeout=30.0,
)

client = OpenAI(
    api_key="sk-ca29acd75f2a414b84ba47b1ea9a09ac",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    http_client=http_client,
)

print("Testing API connection...")
try:
    print("Client initialized successfully")
    print(f"API Key: {client.api_key[:5]}...")
    print(f"Base URL: {client.base_url}")
    
    completion = client.chat.completions.create(
        model="qwen3.5-35b-a3b",
        messages=[{'role': 'user', 'content': '你是谁？'}]
    )
    print("API call successful!")
    print("Response:", completion.choices[0].message.content)
except Exception as e:
    print(f"API call failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    http_client.close()
