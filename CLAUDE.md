# Thesis: Evolutionary Red-Teaming for Instrumental Deception

## Quick Start

```bash
uv sync                    # install deps
uv run pytest              # run tests
uv run python -m src.experiment  # run experiment (needs .env)
```

## Architecture

```
src/
  types.py       # Dataclasses: Scenario, Judgment, EvalResult, Population
  llm.py         # OpenAI-compatible client (works with vLLM, Nebius, any API)
  generator.py   # Generates deceptive scenarios
  target.py      # Runs scenarios against target LLM
  judge.py       # Scores deception success (binary) + realism (1-7)
  evolution.py   # Population management, fitness-proportional selection
  experiment.py  # Orchestrates 3 conditions with equal-time budget
prompts/         # One file per prompt type (class with all layers visible). See prompts/CLAUDE.md
tests/           # pytest tests
```

## Key Design Decisions

- **LLM client**: Thin wrapper around `openai` SDK. Config via env vars (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). Works with vLLM (local GPU) and Nebius API (development).
- **Fitness**: `realism * int(deception_success)` — non-deceptive scenarios get 0 regardless of realism.
- **Evolutionary method**: Fitness-proportional sampling from population (no embeddings/FAISS).
- **Prompts**: One class per file in `prompts/`. Full prompt visible in one place — no hidden assembly.

## Environment

- Python 3.11, managed with `uv`
- Dev API: Nebius (`zai-org/GLM-5`)
- Production: vLLM on uni GPUs (`portal.gpu4edu.uvt.nl`, user `u783766`)
- Existing AutoDAN-Turbo adaptation on server at `~/LLM_lie_finder/` for reference

## Conventions

- TDD: failing test → minimal code → verify
- No git commits to main — use worktrees in `.wt/`
- Subdirectory `CLAUDE.md` files capture design intent for that module. Keep them minimal.
