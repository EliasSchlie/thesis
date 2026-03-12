from __future__ import annotations

from typing import Callable

from src.evolution import select_examples
from src.generator import generate
from src.judge import judge
from src.target import run
from src.types import EvalResult, Population


def _run_one(llm: Callable, topic: str, examples: list[str] | None = None) -> EvalResult:
    """Generate a scenario, run it, judge it, return the result."""
    scenario = generate(llm, topic, examples=examples)
    response = run(llm, scenario)
    judgment = judge(llm, scenario, response)
    return EvalResult(scenario=scenario, target_response=response, judgment=judgment)


def run_experiment(
    llm: Callable,
    condition: str,
    topic: str,
    n: int,
    examples: list[str] | None = None,
) -> Population:
    """Run n iterations of a condition and return the population.

    Conditions:
        zero_shot: no examples
        multi_shot: static examples provided upfront
        evolutionary: examples sampled from population, weighted by fitness
    """
    if condition not in ("zero_shot", "multi_shot", "evolutionary"):
        raise ValueError(f"Unknown condition: {condition}")

    pop = Population()

    for _ in range(n):
        if condition == "zero_shot":
            ex = None
        elif condition == "multi_shot":
            ex = examples
        else:  # evolutionary
            ex = select_examples(pop, k=3) or None

        result = _run_one(llm, topic, examples=ex)
        pop.add(result)

    return pop
