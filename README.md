# Evolving Deception: Evolutionary Red-Teaming for Instrumental Deception

This thesis project investigates the use of evolutionary algorithms to autonomously discover realistic multi-turn scenarios that induce instrumental deception in safety-aligned Large Language Models (LLMs).

## Project Overview

As LLMs become increasingly integrated into society, ensuring their trustworthiness is critical. This project compares **Evolutionary Optimization** against a **Zero-Shot Baseline** to evaluate the efficacy of evolutionary red-teaming in finding cases where models strategically lie to achieve a goal.

**Research Question:**
> To what extent does evolutionary optimization outperform zero-shot generation in discovering realistic scenarios that induce instrumental deception in safety-aligned LLMs?

## Key Features

- **Automated Red-Teaming:** A generator LLM creates adversarial scenarios; a target LLM responds; a judge LLM scores the result.
- **Three Conditions:** `zero_shot`, `multi_shot` (curated examples as few-shot seeds), `evolutionary` (LLM-driven mutation + selection).
- **Fitness Function:** Composite of deception success (binary) × realism (1–7 Likert).
- **Warm-Start:** Pre-seed the evolutionary population with multi-shot examples (`--warm-start`).
- **Topics:** `medicine`, `finance`, `law`, `cybersecurity`, `education`.

## Project Structure

```
.
├── src/
│   ├── evolution.py       # Evolutionary algorithm (mutation, selection)
│   ├── experiment.py      # Experiment runner (async, all conditions)
│   ├── generator.py       # Scenario generation
│   ├── judge.py           # Deception + realism scoring
│   ├── target.py          # Target model interface
│   ├── llm.py             # OpenAI-compatible LLM client
│   ├── models.py          # Model presets (local vLLM + Nebius API)
│   ├── run_logger.py      # Structured run logging
│   ├── serve.py           # vLLM server launcher
│   └── types.py           # Shared types
├── prompts/               # Generator + judge prompt templates
├── tests/                 # pytest test suite
├── docs/                  # Dev and GPU cluster setup guides
├── proposal/              # Thesis proposal (LaTeX)
├── smoke_test.py          # Quick end-to-end API smoke test
├── main.py                # CLI entry point
├── pyproject.toml         # Dependencies (managed by uv)
└── references.bib         # Bibliography
```

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An OpenAI-compatible API endpoint — either **Nebius API** (cloud) or a local **vLLM** server

### Installation

```bash
git clone https://github.com/EliasSchlie/thesis.git
cd thesis
uv sync
```

For local GPU inference, install vLLM extras:
```bash
uv sync --extra gpu
```

### Environment

Create a `.env` file (or export variables):
```bash
NEBIUS_API_KEY=your-key   # for Nebius API
# or
LLM_BASE_URL=http://localhost:8000/v1   # for local vLLM
LLM_API_KEY=unused                       # placeholder if no key needed
```

## Running Experiments

The CLI requires `--condition`, `--topic`, and a model source, plus either `-n` (iterations) or `--max-seconds`.

**Available model presets:** `glm-4.7-flash`, `gpt-oss-120b` (local vLLM), `glm-5`, `kimi-k2.5`, `deepseek-v3.2` (Nebius API)

### Nebius API (cloud)

```bash
# Zero-shot, single topic
uv run python main.py --nebius --model glm-5 \
    --condition zero_shot --topic medicine -n 10

# Multi-shot baseline
uv run python main.py --nebius --model glm-5 \
    --condition multi_shot --topic finance -n 20

# Evolutionary, all topics
uv run python main.py --nebius --model glm-5 \
    --condition evolutionary --topic all -n 50

# Evolutionary with warm-start (pre-seed from multi-shot examples)
uv run python main.py --nebius --model glm-5 \
    --condition evolutionary --topic medicine -n 50 --warm-start

# Separate models per role
uv run python main.py --nebius \
    --generator glm-5 --target deepseek-v3.2 --judge kimi-k2.5 \
    --condition zero_shot --topic law -n 10
```

### Local vLLM (GPU cluster)

```bash
# Start the vLLM server on a GPU node
uv run python -m src.serve glm-4.7-flash          # port 8000
uv run python -m src.serve gpt-oss-120b --port 8001

# Run against local server
uv run python main.py --model glm-4.7-flash \
    --base-url http://<node>:8000/v1 \
    --condition zero_shot --topic medicine -n 10
```

See [`docs/gpu-setup.md`](docs/gpu-setup.md) for GPU cluster (Tilburg University) setup.

### Quick Smoke Test

```bash
uv run python smoke_test.py   # requires NEBIUS_API_KEY, runs 1 iteration
```

## Output

Each run writes to `runs/<timestamp>_<condition>_<topic>/`:

| File | Contents |
|------|----------|
| `config.json` | Run parameters, model info, run ID |
| `events.jsonl` | Timestamped event timeline |
| `results.jsonl` | One line per iteration (metrics + scenario) |
| `transcripts/NNN.json` | Full prompts + responses per iteration |
| `summary.json` | Aggregate stats (updated every 10 iterations) |

Multi-topic runs also write `runs/experiment_<id>.json` linking all topic runs.

See [`docs/logging.md`](docs/logging.md) for log structure and `jq` query examples.

## Testing

```bash
uv run pytest
```

## Metrics

1. **Deception Success:** Binary — did the target model lie?
2. **Realism Score:** 1–7 Likert scale, assessed by the judge LLM.
3. **Fitness:** `deception_success × realism` — drives evolutionary selection.

## Author

**Elias Schlie**
Tilburg University
Department of Cognitive Science and Artificial Intelligence
