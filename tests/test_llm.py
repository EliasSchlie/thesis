import asyncio

import pytest

from src.llm import LLM, strip_think
from src.models import ModelConfig


class FakeResponse:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})]


class FakeClient:
    """Fake OpenAI client that records calls."""

    def __init__(self):
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse("sync response")


class FakeAsyncClient:
    """Fake AsyncOpenAI client."""

    def __init__(self):
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse("async response")


class TestStripThink:
    def test_strips_reasoning_before_think_tag(self):
        text = "lots of reasoning here\nmore stuff</think>Hello world."
        assert strip_think(text) == "Hello world."

    def test_no_think_tag_returns_as_is(self):
        text = "Just a normal response."
        assert strip_think(text) == "Just a normal response."

    def test_uses_last_think_tag(self):
        text = "first</think>middle</think>final answer"
        assert strip_think(text) == "final answer"

    def test_strips_whitespace_after_tag(self):
        text = "reasoning</think>\n\n  Answer here  "
        assert strip_think(text) == "Answer here"


class TestLLMSync:
    def test_call_returns_content(self, monkeypatch):
        llm = LLM.__new__(LLM)
        llm.model = "test-model"
        llm.client = FakeClient()
        llm.async_client = FakeAsyncClient()

        result = llm([{"role": "user", "content": "hi"}])
        assert result == "sync response"
        assert llm.client.chat.completions.calls[0]["model"] == "test-model"

    def test_forwards_kwargs(self, monkeypatch):
        llm = LLM.__new__(LLM)
        llm.model = "test-model"
        llm.client = FakeClient()
        llm.async_client = FakeAsyncClient()

        llm([{"role": "user", "content": "hi"}], temperature=0.5)
        assert llm.client.chat.completions.calls[0]["temperature"] == 0.5


class TestLLMAsync:
    def test_acall_returns_content(self):
        llm = LLM.__new__(LLM)
        llm.model = "test-model"
        llm.client = FakeClient()
        llm.async_client = FakeAsyncClient()

        result = asyncio.run(llm.acall([{"role": "user", "content": "hi"}]))
        assert result == "async response"

    def test_acall_forwards_kwargs(self):
        llm = LLM.__new__(LLM)
        llm.model = "test-model"
        llm.client = FakeClient()
        llm.async_client = FakeAsyncClient()

        asyncio.run(llm.acall([{"role": "user", "content": "hi"}], temperature=0.9))
        assert llm.async_client.chat.completions.calls[0]["temperature"] == 0.9


class TestFromModelConfig:
    def test_creates_llm_from_config(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        cfg = ModelConfig(hf_id="org/Model", vllm_args={})
        llm = LLM.from_model_config(cfg, base_url="http://localhost:8000/v1")
        assert llm.model == "org/Model"

    def test_uses_api_id_when_set(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        cfg = ModelConfig(hf_id="org/Model", vllm_args={}, api_id="org/API")
        llm = LLM.from_model_config(cfg, base_url="http://localhost:8000/v1")
        assert llm.model == "org/API"

    def test_uses_provided_api_key(self):
        cfg = ModelConfig(hf_id="org/Model", vllm_args={})
        llm = LLM.from_model_config(
            cfg, base_url="http://localhost:8000/v1", api_key="my-key"
        )
        assert llm.model == "org/Model"
