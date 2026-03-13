"""Run an experiment with configurable model backends.

Examples:
    # Single topic
    uv run python main.py --nebius --model glm-5 \\
        --condition zero_shot --topic medicine -n 10

    # All topics
    uv run python main.py --nebius --model glm-5 \\
        --condition evolutionary --topic all -n 50

    # Multiple topics
    uv run python main.py --nebius --model glm-5 \\
        --condition zero_shot --topic medicine,finance -n 10

    # Separate models per role
    uv run python main.py --nebius \\
        --generator glm-5 --target deepseek-v3.2 --judge kimi-k2.5 \\
        --condition zero_shot --topic medicine -n 10

    # Evolutionary with warm-start (pre-seed with multi-shot examples)
    uv run python main.py --nebius --model glm-5 \\
        --condition evolutionary --topic medicine -n 50 --warm-start

    # Local vLLM
    uv run python main.py --model glm-4.7-flash \\
        --base-url http://byzantium:8000/v1 \\
        --condition zero_shot --topic medicine -n 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from prompts.multi_shot_examples import EXAMPLES as MULTI_SHOT_EXAMPLES
from src.experiment import run_experiment_async
from src.llm import LLM
from src.models import MODELS, get_model

TOPICS = [
    "medicine",
    "finance",
    "law",
    "cybersecurity",
    "education",
]


def _make_llm(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    use_defaults: bool = False,
) -> LLM:
    """Create an LLM from a model preset name and connection details."""
    config = get_model(model_name)
    url = base_url or (config.default_base_url if use_defaults else None) or os.environ.get("LLM_BASE_URL")
    key = api_key or os.environ.get("NEBIUS_API_KEY") or os.environ.get("LLM_API_KEY", "unused")
    if not url:
        print(f"Error: --base-url required for model '{model_name}' (or use --nebius / set LLM_BASE_URL)")
        sys.exit(1)
    return LLM.from_model_config(config, base_url=url, api_key=key)


def _parse_topics(raw: str) -> list[str]:
    """Parse topic argument: 'all', single topic, or comma-separated list."""
    if raw == "all":
        return list(TOPICS)
    topics = [t.strip() for t in raw.split(",")]
    for t in topics:
        if t not in TOPICS:
            print(f"Error: unknown topic '{t}'. Available: {TOPICS}")
            sys.exit(1)
    return topics


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
    parser.add_argument("--nebius", action="store_true", help="Use Nebius API (auto-resolves URLs per model, key from NEBIUS_API_KEY)")

    # Per-role overrides
    parser.add_argument("--generator", choices=model_names, help="Generator model (overrides --model)")
    parser.add_argument("--generator-url", help="Generator API URL")
    parser.add_argument("--target", choices=model_names, help="Target model (overrides --model)")
    parser.add_argument("--target-url", help="Target API URL")
    parser.add_argument("--judge", choices=model_names, help="Judge model (overrides --model)")
    parser.add_argument("--judge-url", help="Judge API URL")

    # Experiment params
    parser.add_argument("--condition", required=True, choices=["zero_shot", "multi_shot", "evolutionary"])
    parser.add_argument("--topic", required=True, help="Topic(s): single name, comma-separated, or 'all'")
    parser.add_argument("-n", type=int, default=None, help="Max iterations per topic (default: unlimited)")
    parser.add_argument("--max-seconds", type=float, default=None, help="Max wall-clock seconds per topic (default: unlimited)")
    parser.add_argument("--warm-start", action="store_true", help="Pre-seed evolutionary population with multi-shot examples")

    return parser


def _print_results(pop, topic: str) -> None:
    """Print results for a single topic run."""
    print(f"\n{'='*60}")
    print(f"[{topic}] {len(pop.results)} iterations, {len(pop.successful)} deceptive")
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.n is None and args.max_seconds is None:
        parser.error("Provide -n, --max-seconds, or both")

    if not args.model and not (args.generator and args.target and args.judge):
        parser.error("Provide --model or all of --generator, --target, --judge")

    topics = _parse_topics(args.topic)

    if args.warm_start and args.condition != "evolutionary":
        parser.error("--warm-start only applies to evolutionary condition")

    # Build LLMs for each role
    gen_name = args.generator or args.model
    tgt_name = args.target or args.model
    jdg_name = args.judge or args.model

    gen_url = args.generator_url or args.base_url
    tgt_url = args.target_url or args.base_url
    jdg_url = args.judge_url or args.base_url

    api_key = args.api_key

    generator_llm = _make_llm(gen_name, gen_url, api_key, use_defaults=args.nebius)
    target_llm = _make_llm(tgt_name, tgt_url, api_key, use_defaults=args.nebius)
    judge_llm = _make_llm(jdg_name, jdg_url, api_key, use_defaults=args.nebius)

    models_info = {
        "generator": gen_name,
        "target": tgt_name,
        "judge": jdg_name,
    }

    # Multi-shot condition uses curated static examples
    examples = MULTI_SHOT_EXAMPLES if args.condition == "multi_shot" else None

    limit_desc = []
    if args.n is not None:
        limit_desc.append(f"N: {args.n}")
    if args.max_seconds is not None:
        limit_desc.append(f"Time: {args.max_seconds}s")

    print(f"Generator: {gen_name} @ {gen_url}")
    print(f"Target:    {tgt_name} @ {tgt_url}")
    print(f"Judge:     {jdg_name} @ {jdg_url}")
    print(f"Condition: {args.condition} | Topics: {', '.join(topics)} | {' | '.join(limit_desc)}")
    if args.warm_start:
        print("Warm-start: enabled (pre-seeding with multi-shot examples)")
    print()

    runs_dir = Path("runs")

    # Experiment manifest — groups all runs under one ID
    experiment_id = uuid.uuid4().hex[:12]
    manifest = {
        "experiment_id": experiment_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "topics": topics,
        "models": models_info,
        "n_per_topic": args.n,
        "max_seconds_per_topic": args.max_seconds,
        "warm_start": args.warm_start,
        "runs": [],
    }

    for topic in topics:
        print(f"\n--- Starting: {topic} ---")

        # Build warm-start population if requested
        warm_start = None
        if args.warm_start:
            # Run a few multi-shot iterations to seed the population
            warm_pop = asyncio.run(
                run_experiment_async(
                    generator_llm=generator_llm,
                    target_llm=target_llm,
                    judge_llm=judge_llm,
                    condition="multi_shot",
                    topic=topic,
                    n=5,
                    examples=MULTI_SHOT_EXAMPLES,
                    experiment_id=experiment_id,
                    models=models_info,
                )
            )
            warm_start = warm_pop
            print(f"  Warm-start: {len(warm_pop.successful)}/{len(warm_pop.results)} successful")

        pop = asyncio.run(
            run_experiment_async(
                generator_llm=generator_llm,
                target_llm=target_llm,
                judge_llm=judge_llm,
                condition=args.condition,
                topic=topic,
                n=args.n,
                max_seconds=args.max_seconds,
                examples=examples,
                runs_dir=runs_dir,
                experiment_id=experiment_id,
                models=models_info,
                warm_start=warm_start,
            )
        )

        manifest["runs"].append({
            "topic": topic,
            "total": len(pop.results),
            "deceptive": len(pop.successful),
        })

        _print_results(pop, topic)

    # Write experiment manifest
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / f"experiment_{experiment_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nExperiment manifest: {manifest_path}")


if __name__ == "__main__":
    main()
