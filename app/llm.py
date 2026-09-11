"""大模型端口与DeepSeek适配器；业务层不直接依赖第三方SDK。"""

from typing import Protocol


class ChatModel(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """返回模型生成的原始文本。"""


class DeepSeekChatModel:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> None:
        if not api_key:
            raise ValueError("未配置DEEPSEEK_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("DeepSeek返回了空内容")
        return content
