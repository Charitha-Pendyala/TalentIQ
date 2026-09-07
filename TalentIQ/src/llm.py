"""
DeepSeek LLM client wrapper.

DeepSeek's API is OpenAI-compatible, so we talk to it with plain `requests`
calls rather than pulling in the full openai SDK. This keeps the dependency
footprint small and makes it easy to swap providers later if needed.
"""

import os
import requests
from typing import List, Dict


class DeepSeekLLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Copy .env.example to .env and add your key."
            )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        Returns the assistant's text response.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
