from src.models import MODELS, ModelConfig, get_model


class TestModelConfig:
    def test_has_required_fields(self):
        cfg = ModelConfig(
            hf_id="org/Model",
            vllm_args={"tensor_parallel_size": 2},
        )
        assert cfg.hf_id == "org/Model"
        assert cfg.vllm_args == {"tensor_parallel_size": 2}
        assert cfg.api_id is None

    def test_api_id_overrides_hf_id(self):
        cfg = ModelConfig(
            hf_id="org/Model",
            vllm_args={},
            api_id="org/Model-API",
        )
        assert cfg.api_id == "org/Model-API"

    def test_model_id_returns_api_id_when_set(self):
        cfg = ModelConfig(hf_id="org/Model", vllm_args={}, api_id="org/API")
        assert cfg.model_id == "org/API"

    def test_model_id_returns_hf_id_when_no_api_id(self):
        cfg = ModelConfig(hf_id="org/Model", vllm_args={})
        assert cfg.model_id == "org/Model"


class TestGetModel:
    def test_returns_known_model(self):
        cfg = get_model("glm-4.7-flash")
        assert cfg.hf_id == "zai-org/GLM-4.7-Flash"

    def test_returns_gpt_oss(self):
        cfg = get_model("gpt-oss-120b")
        assert cfg.hf_id == "openai/gpt-oss-120b"

    def test_unknown_model_raises(self):
        import pytest

        with pytest.raises(KeyError, match="unknown-model"):
            get_model("unknown-model")


class TestRegistry:
    def test_glm_has_trust_remote_code(self):
        cfg = MODELS["glm-4.7-flash"]
        assert cfg.vllm_args["trust_remote_code"] is True

    def test_gpt_oss_has_enforce_eager(self):
        cfg = MODELS["gpt-oss-120b"]
        assert cfg.vllm_args["enforce_eager"] is True

    def test_vllm_models_have_tensor_parallel(self):
        for name, cfg in MODELS.items():
            if cfg.vllm_args:
                assert "tensor_parallel_size" in cfg.vllm_args, f"{name} missing TP"

    def test_api_models_have_api_id(self):
        api_models = ["glm-5", "kimi-k2.5", "deepseek-v3.2"]
        for name in api_models:
            cfg = MODELS[name]
            assert cfg.api_id is not None, f"{name} missing api_id"
            assert cfg.vllm_args == {}, f"{name} should have no vllm_args"
