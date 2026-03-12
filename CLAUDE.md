# Thesis: Evolutionary Red-Teaming for Instrumental Deception

## Quick Start

```bash
uv sync && uv run pytest
```

Setup: [docs/dev-setup.md](docs/dev-setup.md) (local API) | [docs/gpu-setup.md](docs/gpu-setup.md) (uni GPUs + vLLM)

## Architecture

```
src/
  types.py       # Scenario, Judgment, EvalResult, Population
  llm.py         # OpenAI-compatible client (sync + async, strips </think>)
  models.py      # Model registry with vLLM presets
  serve.py       # Launch vLLM server for a registered model
  generator.py   # Generates deceptive scenarios
  target.py      # Runs scenarios against target LLM
  judge.py       # Deception success (binary) + realism (1-7), async-parallel
  evolution.py   # Fitness-proportional selection from population
  experiment.py  # Orchestrates 3 conditions, per-role LLM support
  run_logger.py  # Structured logging: events, results, transcripts per run
prompts/         # One class per file. See prompts/CLAUDE.md
tests/
main.py          # CLI entry point
```

## Key Design Decisions

- **LLM = callable**: `llm(messages, **kwargs) -> str`. Pipeline components don't know about APIs/GPUs.
- **Per-role LLMs**: Generator, target, and judge can use different models/servers.
- **Fitness**: `realism × deception_success` — non-deceptive → 0 regardless of realism.
- **Evolutionary**: Fitness-proportional sampling from population (no embeddings/FAISS).
- **Prompts**: One class per file in `prompts/`. Full prompt visible in one place.
- **Logging**: Each run creates `runs/<ts>_<condition>_<topic>/` with events.jsonl, results.jsonl, transcripts/, summary.json. See [docs/logging.md](docs/logging.md).

## Conventions

- TDD: failing test → minimal code → verify
- No git commits to main — use worktrees in `.wt/`
- Subdirectory `CLAUDE.md` files capture design intent. Keep them minimal.
