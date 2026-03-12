# Local Development Setup

## Install

```bash
uv sync          # no GPU deps needed
uv run pytest    # verify
```

## Running with Nebius API

```bash
# Set up .env
echo "NEBIUS_API_KEY=your-key" > .env

# Run via smoke test
uv run python smoke_test.py

# Run via CLI
uv run python main.py --model glm-4.7-flash \
    --base-url https://api.tokenfactory.us-central1.nebius.com/v1/ \
    --api-key $NEBIUS_API_KEY \
    --condition zero_shot --topic medicine -n 5
```

Note: `--model` selects the model preset (for its HF ID). The `--base-url` determines where the request actually goes. Any OpenAI-compatible API works.
