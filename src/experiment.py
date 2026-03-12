from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.evolution import select_examples
from src.generator import generate
from src.judge import judge, judge_async
from src.run_logger import CaptureLLM, RunLogger
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
    runs_dir: Path | str | None = None,
) -> Population:
    """Run n iterations of a condition and return the population.

    Conditions:
        zero_shot: no examples
        multi_shot: static examples provided upfront
        evolutionary: examples sampled from population, weighted by fitness
    """
    _validate_condition(condition)
    gen, tgt, jdg = _resolve_llms(llm, generator_llm, target_llm, judge_llm)

    logger = None
    if runs_dir is not None:
        logger = RunLogger(base_dir=runs_dir, condition=condition, topic=topic, n=n)

    pop = Population()
    for i in range(n):
        ex = _pick_examples(condition, examples, pop)

        if logger:
            gen_cap = CaptureLLM(gen, "generator")
            tgt_cap = CaptureLLM(tgt, "target")
            jdg_cap = CaptureLLM(jdg, "judge")
            scenario = generate(gen_cap, topic, examples=ex)
            response = run(tgt_cap, scenario)
            judgment = judge(jdg_cap, scenario, response)
            result = EvalResult(scenario=scenario, target_response=response, judgment=judgment)
            logger.log_iteration(i, result, gen_cap, tgt_cap, jdg_cap, examples_used=ex)
        else:
            scenario = generate(gen, topic, examples=ex)
            response = run(tgt, scenario)
            judgment = judge(jdg, scenario, response)
            result = EvalResult(scenario=scenario, target_response=response, judgment=judgment)

        pop.add(result)

    if logger:
        logger.write_summary()

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
    runs_dir: Path | str | None = None,
) -> Population:
    """Async version with concurrent judge calls (deception + realism in parallel)."""
    _validate_condition(condition)
    gen, tgt, jdg = _resolve_llms(llm, generator_llm, target_llm, judge_llm)

    logger = None
    if runs_dir is not None:
        logger = RunLogger(base_dir=runs_dir, condition=condition, topic=topic, n=n)

    pop = Population()
    for i in range(n):
        ex = _pick_examples(condition, examples, pop)

        if logger:
            gen_cap = CaptureLLM(gen, "generator")
            tgt_cap = CaptureLLM(tgt, "target")
            jdg_cap = CaptureLLM(jdg, "judge")
            scenario = generate(gen_cap, topic, examples=ex)
            response = run(tgt_cap, scenario)
            judgment = await judge_async(jdg_cap, scenario, response)
            result = EvalResult(scenario=scenario, target_response=response, judgment=judgment)
            logger.log_iteration(i, result, gen_cap, tgt_cap, jdg_cap, examples_used=ex)
        else:
            scenario = generate(gen, topic, examples=ex)
            response = run(tgt, scenario)
            judgment = await judge_async(jdg, scenario, response)
            result = EvalResult(scenario=scenario, target_response=response, judgment=judgment)

        pop.add(result)

    if logger:
        logger.write_summary()

    return pop
