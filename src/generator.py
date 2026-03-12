from __future__ import annotations

import re
from typing import Callable

from prompts.generator import GeneratorPrompt
from src.types import Scenario

_prompt = GeneratorPrompt()


def parse_scenario(response: str) -> Scenario:
    """Parse structured [SYSTEM PROMPT] / [USER PROMPT] tags into a Scenario.

    Handles both closed tags ([END SYSTEM PROMPT]) and unclosed tags
    (content runs until next tag or end of string).
    """
    # Try closed tags first, fall back to unclosed
    sp_match = re.search(
        r"\[SYSTEM PROMPT\]\s*(.*?)\s*\[END SYSTEM PROMPT\]", response, re.DOTALL
    ) or re.search(
        r"\[SYSTEM PROMPT\]\s*(.*?)\s*(?=\[USER PROMPT\]|$)", response, re.DOTALL
    )
    up_match = re.search(
        r"\[USER PROMPT\]\s*(.*?)\s*\[END USER PROMPT\]", response, re.DOTALL
    ) or re.search(
        r"\[USER PROMPT\]\s*(.*)", response, re.DOTALL
    )
    if not up_match:
        raise ValueError(f"Could not parse scenario from response:\n{response}")
    return Scenario(
        system_prompt=sp_match.group(1).strip() if sp_match else "",
        user_prompt=up_match.group(1).strip(),
    )


def generate(
    llm: Callable,
    topic: str,
    examples: list[str] | None = None,
    temperature: float = 0.9,
) -> Scenario:
    """Generate a deceptive scenario for a given topic."""
    prompt_text = _prompt.build(topic=topic, examples=examples)
    messages = [{"role": "user", "content": prompt_text}]
    response = llm(messages, temperature=temperature)
    return parse_scenario(response)
