from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable

from src.evolution import select_examples
from src.generator import generate
from src.judge import judge_async
from src.run_logger import CaptureLLM, RunLogger
from src.target import run
from src.types import EvalResult, Population

logger = logging.getLogger(__name__)


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
    experiment_id: str | None = None,
    models: dict[str, str] | None = None,
    warm_start: Population | None = None,
) -> Population:
    """Run iterations of a condition and return the population.

    Stops when either n iterations complete or max_seconds elapses,
    whichever comes first. At least one of n or max_seconds must be set.

    Conditions:
        zero_shot: no examples
        multi_shot: static examples provided upfront
        evolutionary: examples sampled from population, weighted by fitness

    warm_start: Pre-seed the population (e.g. with multi-shot results so
        evolutionary selection has candidates from iteration 0).
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
            experiment_id=experiment_id,
            models=models,
            warm_start=warm_start,
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
    experiment_id: str | None = None,
    models: dict[str, str] | None = None,
    warm_start: Population | None = None,
) -> Population:
    """Run iterations with concurrent judge calls (deception + realism in parallel).

    Stops when either n iterations complete or max_seconds elapses,
    whichever comes first. At least one of n or max_seconds must be set.
    """
    if n is None and max_seconds is None:
        raise ValueError("Must provide n, max_seconds, or both")
    _validate_condition(condition)
    gen, tgt, jdg = _resolve_llms(llm, generator_llm, target_llm, judge_llm)

    run_logger = None
    if runs_dir is not None:
        run_logger = RunLogger(
            base_dir=runs_dir, condition=condition, topic=topic,
            n=n, max_seconds=max_seconds,
            experiment_id=experiment_id, models=models,
        )

    pop = Population()
    if warm_start:
        for result in warm_start.results:
            pop.add(result)
        if run_logger:
            run_logger.event(
                "warm_start",
                count=len(warm_start.results),
                successful=len(warm_start.successful),
            )

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

        elapsed = time.monotonic() - start
        try:
            scenario = generate(gen_cap, topic, examples=ex)
            response = run(tgt_cap, scenario)
            judgment = await judge_async(jdg_cap, scenario, response)
        except Exception as e:
            logger.warning("Iteration %d failed: %s", i, e)
            if run_logger:
                run_logger.log_error(i, str(e), elapsed_seconds=elapsed)
                run_logger.log_transcript_from_captures(
                    i, gen_cap.drain(), tgt_cap.drain(), jdg_cap.drain(),
                )
            i += 1
            continue

        result = EvalResult(scenario=scenario, target_response=response, judgment=judgment)

        elapsed = time.monotonic() - start
        if run_logger:
            run_logger.log_iteration(
                i, result, gen_cap, tgt_cap, jdg_cap,
                examples_used=ex, elapsed_seconds=elapsed,
            )

        pop.add(result)
        i += 1

    if run_logger:
        run_logger.write_summary(elapsed_seconds=time.monotonic() - start)

    return pop
