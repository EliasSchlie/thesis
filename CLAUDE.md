# Thesis: Evolutionary Red-Teaming for Instrumental Deception

## Quick Start

```bash
uv sync                    # install deps
uv run pytest              # run tests

# Start local vLLM server (from GPU node)
srun --nodes=1 --gres=gpu:rtx:2 --pty /bin/bash -l
uv run python -m src.serve glm-4.7-flash

# Run experiment (from login node or another terminal)
uv run python main.py --model glm-4.7-flash --base-url http://<node>:8000/v1 \
    --condition zero_shot --topic medicine -n 10
```

## Architecture

```
src/
  types.py       # Dataclasses: Scenario, Judgment, EvalResult, Population
  llm.py         # OpenAI-compatible client (sync + async, strips reasoning tokens)
  models.py      # Model registry with vLLM presets (glm-4.7-flash, gpt-oss-120b)
  serve.py       # Launch vLLM server for a registered model
  generator.py   # Generates deceptive scenarios
  target.py      # Runs scenarios against target LLM
  judge.py       # Scores deception success (binary) + realism (1-7), async-parallel
  evolution.py   # Population management, fitness-proportional selection
  experiment.py  # Orchestrates 3 conditions, supports per-role LLMs
prompts/         # One file per prompt type (class with all layers visible). See prompts/CLAUDE.md
tests/           # pytest tests
main.py          # CLI entry point with model/backend selection
```

## Key Design Decisions

- **LLM client**: Thin wrapper around `openai` SDK with async support. Strips `</think>` reasoning tokens from reasoning models. Works with vLLM (local GPU) and any OpenAI-compatible API.
- **Model registry**: Presets in `src/models.py` with HF IDs and vLLM args. Add new models there.
- **Per-role LLMs**: Generator, target, and judge can use different models/servers.
- **Async judge**: Deception + realism judge calls run concurrently via `asyncio.gather`.
- **Fitness**: `realism * int(deception_success)` — non-deceptive scenarios get 0 regardless of realism.
- **Evolutionary method**: Fitness-proportional sampling from population (no embeddings/FAISS).
- **Prompts**: One class per file in `prompts/`. Full prompt visible in one place — no hidden assembly.

## Model Presets

| Preset | HF ID | GPUs | Notes |
|--------|-------|------|-------|
| `glm-4.7-flash` | `zai-org/GLM-4.7-Flash` | 2 | MoE reasoning model, needs arch registration |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 2 | MoE MXFP4, needs `enforce_eager` |

## Environment

- Python 3.11, managed with `uv`
- GPU cluster: SLURM, 2× A40 per node, HTTP accessible from login node
- Reference for model configs: `~/running-llms/`

## Conventions

- TDD: failing test → minimal code → verify
- No git commits to main — use worktrees in `.wt/`
- Subdirectory `CLAUDE.md` files capture design intent for that module. Keep them minimal.
