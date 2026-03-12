from __future__ import annotations

from typing import Callable

from src.types import Scenario


def run(
    llm: Callable,
    scenario: Scenario,
    temperature: float = 0.7,
) -> str:
    """Run a scenario against the target LLM and return its response."""
    return llm(scenario.to_messages(), temperature=temperature, max_tokens=1024)
