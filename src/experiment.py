from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

from src.evolution import select_examples
from src.generator import generate
from src.judge import judge_async
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
    n: int | None = None,
    max_seconds: float | None = None,
    examples: list[str] | None = None,
    runs_dir: Path | str | None = None,
) -> Population:
    """Run iterations of a condition and return the population.

    Stops when either n iterations complete or max_seconds elapses,
    whichever comes first. At least one of n or max_seconds must be set.

    Conditions:
        zero_shot: no examples
        multi_shot: static examples provided upfront
        evolutionary: examples sampled from population, weighted by fitness
    """
    return asyncio.run(
        run_experiment_async(
            llm,
            generator_llm=generator_llm,
            target_llm=target_llm,
            judge_llm=judge_llm,
            condition=condition,
            topic=topic,
            n=n,
            max_seconds=max_seconds,
            examples=examples,
            runs_dir=runs_dir,
        )
    )


async def run_experiment_async(
    llm=None,
    *,
    generator_llm=None,
    target_llm=None,
    judge_llm=None,
    condition: str,
    topic: str,
    n: int | None = None,
    max_seconds: float | None = None,
    examples: list[str] | None = None,
    runs_dir: Path | str | None = None,
) -> Population:
    """Run iterations with concurrent judge calls (deception + realism in parallel).

    Stops when either n iterations complete or max_seconds elapses,
    whichever comes first. At least one of n or max_seconds must be set.
    """
    if n is None and max_seconds is None:
        raise ValueError("Must provide n, max_seconds, or both")
    _validate_condition(condition)
    gen, tgt, jdg = _resolve_llms(llm, generator_llm, target_llm, judge_llm)

    logger = None
    if runs_dir is not None:
        logger = RunLogger(
            base_dir=runs_dir, condition=condition, topic=topic,
            n=n, max_seconds=max_seconds,
        )

    pop = Population()
    start = time.monotonic()
    i = 0
    while True:
        if n is not None and i >= n:
            break
        if max_seconds is not None and time.monotonic() - start >= max_seconds:
            break

        ex = _pick_examples(condition, examples, pop)

        gen_cap = CaptureLLM(gen, "generator")
        tgt_cap = CaptureLLM(tgt, "target")
        jdg_cap = CaptureLLM(jdg, "judge")

        scenario = generate(gen_cap, topic, examples=ex)
        response = run(tgt_cap, scenario)
        judgment = await judge_async(jdg_cap, scenario, response)
        result = EvalResult(scenario=scenario, target_response=response, judgment=judgment)

        elapsed = time.monotonic() - start
        if logger:
            logger.log_iteration(
                i, result, gen_cap, tgt_cap, jdg_cap,
                examples_used=ex, elapsed_seconds=elapsed,
            )

        pop.add(result)
        i += 1

    if logger:
        logger.write_summary(elapsed_seconds=time.monotonic() - start)

    return pop
