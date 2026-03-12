# GPU Cluster Setup

## Server Access

```bash
ssh u783766@portal.gpu4edu.uvt.nl
```

## Install (first time)

```bash
uv sync --extra gpu    # includes vLLM + CUDA deps
```

## Running Experiments

```bash
# 1. Get a GPU node
srun --nodes=1 --gres=gpu:rtx:2 --pty /bin/bash -l

# 2. Start vLLM server
uv run python -m src.serve glm-4.7-flash          # default port 8000
uv run python -m src.serve gpt-oss-120b --port 8001  # second model

# 3. Run experiment (from login node or another terminal)
uv run python main.py --model glm-4.7-flash --base-url http://<node>:8000/v1 \
    --condition zero_shot --topic medicine -n 10

# Separate models per role
uv run python main.py \
    --generator glm-4.7-flash --generator-url http://<node>:8000/v1 \
    --judge gpt-oss-120b --judge-url http://<node>:8001/v1 \
    --condition evolutionary --topic medicine -n 50
```

## Model Presets

| Preset | HF ID | GPUs | Notes |
|--------|-------|------|-------|
| `glm-4.7-flash` | `zai-org/GLM-4.7-Flash` | 2 | MoE reasoning model, needs arch registration |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 2 | MoE MXFP4, needs `enforce_eager` |

Add new models in `src/models.py`.

## Reference

Existing AutoDAN-Turbo adaptation at `~/LLM_lie_finder/` on the server.
Model config examples at `~/running-llms/`.
