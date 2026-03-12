from __future__ import annotations

import os

from openai import AsyncOpenAI, OpenAI

from src.models import ModelConfig


def strip_think(text: str) -> str:
    """Strip reasoning tokens (everything before </think>) from model output."""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    return text[idx + len(marker) :].strip()


class LLM:
    """Call with messages, get text back. Supports sync and async."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.environ["LLM_MODEL"]
        _base_url = base_url or os.environ["LLM_BASE_URL"]
        _api_key = api_key or os.environ["LLM_API_KEY"]
        self.client = OpenAI(base_url=_base_url, api_key=_api_key)
        self.async_client = AsyncOpenAI(base_url=_base_url, api_key=_api_key)

    @classmethod
    def from_model_config(
        cls,
        config: ModelConfig,
        base_url: str,
        api_key: str | None = None,
    ) -> LLM:
        """Create an LLM from a ModelConfig preset."""
        return cls(
            model=config.model_id,
            base_url=base_url,
            api_key=api_key or os.environ.get("LLM_API_KEY", "unused"),
        )

    def __call__(self, messages: list[dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return strip_think(response.choices[0].message.content)

    async def acall(self, messages: list[dict], **kwargs) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return strip_think(response.choices[0].message.content)
