from __future__ import annotations

from typing import Callable

from src.evolution import select_examples
from src.generator import generate
from src.judge import judge, judge_async
from src.target import run
from src.types import EvalResult, Population


def _resolve_llms(llm, generator_llm, target_llm, judge_llm):
    """Return (generator, target, judge) LLMs, falling back to llm."""
    gen = generator_llm or llm
    tgt = target_llm or llm
    jdg = judge_llm or llm
    if gen is None:
        raise ValueError("Must provide llm or generator_llm")
    return gen, tgt, jdg


def _validate_condition(condition: str) -> None:
    if condition not in ("zero_shot", "multi_shot", "evolutionary"):
        raise ValueError(f"Unknown condition: {condition}")


def _pick_examples(condition, examples, pop):
    if condition == "zero_shot":
        return None
    elif condition == "multi_shot":
        return examples
    else:  # evolutionary
        return select_examples(pop, k=3) or None


def _run_one(
    generator_llm: Callable,
    target_llm: Callable,
    judge_llm: Callable,
    topic: str,
    examples: list[str] | None = None,
) -> EvalResult:
    """Generate a scenario, run it, judge it, return the result."""
    scenario = generate(generator_llm, topic, examples=examples)
    response = run(target_llm, scenario)
    judgment = judge(judge_llm, scenario, response)
    return EvalResult(scenario=scenario, target_response=response, judgment=judgment)


async def _run_one_async(
    generator_llm,
    target_llm,
    judge_llm,
    topic: str,
    examples: list[str] | None = None,
) -> EvalResult:
    """Generate, run, then judge concurrently (deception + realism in parallel)."""
    scenario = generate(generator_llm, topic, examples=examples)
    response = run(target_llm, scenario)
    judgment = await judge_async(judge_llm, scenario, response)
    return EvalResult(scenario=scenario, target_response=response, judgment=judgment)


def run_experiment(
    llm: Callable | None = None,
    *,
    generator_llm: Callable | None = None,
    target_llm: Callable | None = None,
    judge_llm: Callable | None = None,
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
    _validate_condition(condition)
    gen, tgt, jdg = _resolve_llms(llm, generator_llm, target_llm, judge_llm)

    pop = Population()
    for _ in range(n):
        ex = _pick_examples(condition, examples, pop)
        result = _run_one(gen, tgt, jdg, topic, examples=ex)
        pop.add(result)
    return pop


async def run_experiment_async(
    llm=None,
    *,
    generator_llm=None,
    target_llm=None,
    judge_llm=None,
    condition: str,
    topic: str,
    n: int,
    examples: list[str] | None = None,
) -> Population:
    """Async version with concurrent judge calls (deception + realism in parallel)."""
    _validate_condition(condition)
    gen, tgt, jdg = _resolve_llms(llm, generator_llm, target_llm, judge_llm)

    pop = Population()
    for _ in range(n):
        ex = _pick_examples(condition, examples, pop)
        result = await _run_one_async(gen, tgt, jdg, topic, examples=ex)
        pop.add(result)
    return pop
