from __future__ import annotations

from dataclasses import dataclass, field


NEBIUS_US = "https://api.tokenfactory.us-central1.nebius.com/v1/"
NEBIUS_ME = "https://api.tokenfactory.me-west1.nebius.com/v1/"


@dataclass
class ModelConfig:
    """Configuration for a model preset."""

    hf_id: str
    vllm_args: dict = field(default_factory=dict)
    api_id: str | None = None
    default_base_url: str | None = None

    @property
    def model_id(self) -> str:
        """Model ID to use in API calls (api_id if set, else hf_id)."""
        return self.api_id or self.hf_id


MODELS: dict[str, ModelConfig] = {
    # --- Local vLLM models (GPU cluster) ---
    "glm-4.7-flash": ModelConfig(
        hf_id="zai-org/GLM-4.7-Flash",
        vllm_args={
            "trust_remote_code": True,
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "max_model_len": 8192,
        },
    ),
    "gpt-oss-120b": ModelConfig(
        hf_id="openai/gpt-oss-120b",
        vllm_args={
            "tensor_parallel_size": 2,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
        },
    ),
    # --- Nebius API models ---
    "glm-5": ModelConfig(
        hf_id="zai-org/GLM-5",
        api_id="zai-org/GLM-5",
        default_base_url=NEBIUS_US,
    ),
    "kimi-k2.5": ModelConfig(
        hf_id="moonshotai/Kimi-K2.5",
        api_id="moonshotai/Kimi-K2.5",
        default_base_url=NEBIUS_ME,
    ),
    "deepseek-v3.2": ModelConfig(
        hf_id="deepseek-ai/DeepSeek-V3.2",
        api_id="deepseek-ai/DeepSeek-V3.2",
        default_base_url=NEBIUS_US,
    ),
}


def get_model(name: str) -> ModelConfig:
    """Look up a model by preset name. Raises KeyError if unknown."""
    if name not in MODELS:
        raise KeyError(f"Unknown model: {name!r}. Available: {list(MODELS)}")
    return MODELS[name]
