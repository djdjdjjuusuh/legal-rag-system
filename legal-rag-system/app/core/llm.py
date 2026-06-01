"""LLM 客户端 - 借鉴 hello-agents 的 MyLLM，适配 DeepSeek API"""
import os
import time
from typing import Optional, List, Dict, Any, Iterator
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError


class LLMClient:
    """统一的 LLM 调用客户端，使用 OpenAI 兼容接口连接 DeepSeek"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "4096"))
        self.max_retries = 3

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=kwargs.get("timeout", 60),
            max_retries=0,  # We handle retries ourselves with backoff
        )

    def _retry_sleep(self, attempt: int):
        time.sleep(min(1.5 ** attempt, 8.0))

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """非流式调用，带自动重试"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                return response.choices[0].message.content or ""
            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    self._retry_sleep(attempt)
            except APIError as e:
                # Don't retry on auth errors or bad requests
                raise
        raise last_error or RuntimeError("LLM invoke failed after retries")

    def stream_invoke(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式调用，带自动重试"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                return  # Success
            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    self._retry_sleep(attempt)
            except APIError as e:
                raise
        raise last_error or RuntimeError("LLM stream_invoke failed after retries")

    def invoke_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """带工具定义的调用，返回完整 response 以便解析 tool_calls"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.model_dump()
