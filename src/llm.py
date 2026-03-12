from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from src.models import ModelConfig


def strip_think(text: str | None) -> str:
    """Strip reasoning tokens (everything before </think>) from model output."""
    if not text:
        return ""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    return text[idx + len(marker) :].strip()


class LLM:
    """Call with messages, get text back. Supports sync and async."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "unused",
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def from_model_config(
        cls,
        config: ModelConfig,
        base_url: str,
        api_key: str = "unused",
    ) -> LLM:
        """Create an LLM from a ModelConfig preset.

        api_key defaults to "unused" for local vLLM servers that don't check it.
        Pass a real key for hosted APIs (Nebius, OpenRouter, etc.).
        """
        return cls(
            model=config.model_id,
            base_url=base_url,
            api_key=api_key,
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
