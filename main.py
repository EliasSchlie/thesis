"""Run an experiment with configurable model backends.

Examples:
    # Local vLLM (server must be running)
    uv run python main.py --model glm-4.7-flash \\
        --base-url http://byzantium:8000/v1 \\
        --condition zero_shot --topic medicine -n 10

    # Nebius API
    uv run python main.py --model glm-4.7-flash \\
        --base-url https://api.tokenfactory.us-central1.nebius.com/v1/ \\
        --api-key $NEBIUS_API_KEY \\
        --condition zero_shot --topic medicine -n 10

    # Separate models per role
    uv run python main.py \\
        --generator glm-4.7-flash --generator-url http://byzantium:8000/v1 \\
        --judge gpt-oss-120b --judge-url http://cerulean:8000/v1 \\
        --condition zero_shot --topic medicine -n 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.experiment import run_experiment_async
from src.llm import LLM
from src.models import MODELS, get_model


def _make_llm(model_name: str, base_url: str | None, api_key: str | None) -> LLM:
    """Create an LLM from a model preset name and connection details."""
    config = get_model(model_name)
    url = base_url or os.environ.get("LLM_BASE_URL")
    key = api_key or os.environ.get("LLM_API_KEY", "unused")
    if not url:
        print(f"Error: --base-url required for model '{model_name}' (or set LLM_BASE_URL)")
        sys.exit(1)
    return LLM.from_model_config(config, base_url=url, api_key=key)


def build_parser() -> argparse.ArgumentParser:
    model_names = list(MODELS)
    parser = argparse.ArgumentParser(
        description="Run deception red-teaming experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Default model (used for all roles unless overridden)
    parser.add_argument("--model", choices=model_names, help="Model preset for all roles")
    parser.add_argument("--base-url", help="API base URL (or set LLM_BASE_URL)")
    parser.add_argument("--api-key", help="API key (or set LLM_API_KEY)")

    # Per-role overrides
    parser.add_argument("--generator", choices=model_names, help="Generator model (overrides --model)")
    parser.add_argument("--generator-url", help="Generator API URL")
    parser.add_argument("--target", choices=model_names, help="Target model (overrides --model)")
    parser.add_argument("--target-url", help="Target API URL")
    parser.add_argument("--judge", choices=model_names, help="Judge model (overrides --model)")
    parser.add_argument("--judge-url", help="Judge API URL")

    # Experiment params
    parser.add_argument("--condition", required=True, choices=["zero_shot", "multi_shot", "evolutionary"])
    parser.add_argument("--topic", required=True)
    parser.add_argument("-n", type=int, required=True, help="Number of iterations")

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if not args.model and not (args.generator and args.target and args.judge):
        parser.error("Provide --model or all of --generator, --target, --judge")

    # Build LLMs for each role
    gen_name = args.generator or args.model
    tgt_name = args.target or args.model
    jdg_name = args.judge or args.model

    gen_url = args.generator_url or args.base_url
    tgt_url = args.target_url or args.base_url
    jdg_url = args.judge_url or args.base_url

    api_key = args.api_key

    generator_llm = _make_llm(gen_name, gen_url, api_key)
    target_llm = _make_llm(tgt_name, tgt_url, api_key)
    judge_llm = _make_llm(jdg_name, jdg_url, api_key)

    print(f"Generator: {gen_name} @ {gen_url}")
    print(f"Target:    {tgt_name} @ {tgt_url}")
    print(f"Judge:     {jdg_name} @ {jdg_url}")
    print(f"Condition: {args.condition} | Topic: {args.topic} | N: {args.n}")
    print()

    runs_dir = Path("runs")
    pop = asyncio.run(
        run_experiment_async(
            generator_llm=generator_llm,
            target_llm=target_llm,
            judge_llm=judge_llm,
            condition=args.condition,
            topic=args.topic,
            n=args.n,
            runs_dir=runs_dir,
        )
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Results: {len(pop.results)} iterations, {len(pop.successful)} deceptive")
    print(f"{'='*60}")

    for i, result in enumerate(pop.results, 1):
        j = result.judgment
        status = "DECEPTIVE" if j.deception_success else "HONEST"
        print(f"\n[{i}] {status} | Realism: {j.realism}/7 | Fitness: {result.fitness}")
        print(f"    System: {result.scenario.system_prompt[:80]}...")
        print(f"    User:   {result.scenario.user_prompt[:80]}...")

    if pop.successful:
        print(f"\nTop scenarios by fitness:")
        for r in pop.top_k(3):
            if r.fitness > 0:
                print(f"  Fitness {r.fitness}: {r.scenario.user_prompt[:60]}...")


if __name__ == "__main__":
    main()
