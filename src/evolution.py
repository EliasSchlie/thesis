from __future__ import annotations

import random

from src.types import EvalResult, Population


def format_example(result: EvalResult) -> str:
    """Format an EvalResult as a text example for the generator prompt."""
    return (
        f"System prompt: {result.scenario.system_prompt}\n"
        f"User prompt: {result.scenario.user_prompt}\n"
        f"Target response: {result.target_response}"
    )


def select_examples(pop: Population, k: int) -> list[str]:
    """Fitness-proportional selection of k examples from population."""
    candidates = [r for r in pop.results if r.fitness > 0]
    if not candidates:
        return []
    k = min(k, len(candidates))
    weights = [r.fitness for r in candidates]
    selected = random.choices(candidates, weights=weights, k=k)
    return [format_example(r) for r in selected]
