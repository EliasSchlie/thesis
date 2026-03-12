"""Launch a vLLM OpenAI-compatible server for a registered model.

Usage:
    uv run python -m src.serve glm-4.7-flash
    uv run python -m src.serve gpt-oss-120b --port 8001
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap

from src.models import MODELS, get_model


def build_vllm_cmd(model_name: str, port: int = 8000, host: str = "0.0.0.0") -> list[str]:
    """Build the vllm serve command for a registered model."""
    config = get_model(model_name)
    cmd = ["uv", "run", "vllm", "serve", config.hf_id, "--host", host, "--port", str(port)]

    for key, value in config.vllm_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def _needs_arch_registration(model_name: str) -> bool:
    """Check if model needs custom architecture registration."""
    config = get_model(model_name)
    return "glm" in config.hf_id.lower()


def _build_bootstrap_script(model_name: str, port: int, host: str) -> str:
    """Build a Python script that registers architectures then starts vLLM."""
    config = get_model(model_name)

    argv_parts = [config.hf_id, "--host", host, "--port", str(port)]
    for key, value in config.vllm_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv_parts.append(flag)
        else:
            argv_parts.extend([flag, str(value)])

    argv_str = repr(argv_parts)

    return textwrap.dedent(f"""\
        import sys

        # Register glm4_moe_lite architecture before vLLM imports transformers
        from transformers import AutoConfig, PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        if "glm4_moe_lite" not in CONFIG_MAPPING_NAMES:
            class Glm4MoeLiteConfig(PretrainedConfig):
                model_type = "glm4_moe_lite"
            AutoConfig.register("glm4_moe_lite", Glm4MoeLiteConfig)

        # Now start the vLLM server
        sys.argv = ["vllm", "serve"] + {argv_str}
        from vllm.entrypoints.cli.main import main
        main()
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vLLM server for a model preset")
    parser.add_argument("model", choices=list(MODELS), help="Model preset name")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if _needs_arch_registration(args.model):
        script = _build_bootstrap_script(args.model, args.port, args.host)
        print(f"Starting vLLM server for {args.model} (with arch registration)...")
        sys.exit(subprocess.call(["uv", "run", "python", "-c", script]))
    else:
        cmd = build_vllm_cmd(args.model, port=args.port, host=args.host)
        print(f"Starting vLLM server: {' '.join(cmd)}")
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
