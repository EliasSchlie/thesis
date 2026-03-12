from src.serve import build_vllm_cmd


class TestBuildVllmCmd:
    def test_glm_includes_trust_remote_code(self):
        cmd = build_vllm_cmd("glm-4.7-flash")
        assert "zai-org/GLM-4.7-Flash" in cmd
        assert "--trust-remote-code" in cmd
        assert "--tensor-parallel-size" in cmd

    def test_gpt_oss_includes_enforce_eager(self):
        cmd = build_vllm_cmd("gpt-oss-120b")
        assert "openai/gpt-oss-120b" in cmd
        assert "--enforce-eager" in cmd

    def test_custom_port(self):
        cmd = build_vllm_cmd("glm-4.7-flash", port=8001)
        idx = cmd.index("--port")
        assert cmd[idx + 1] == "8001"

    def test_starts_with_uv_run_vllm_serve(self):
        cmd = build_vllm_cmd("glm-4.7-flash")
        assert cmd[:4] == ["uv", "run", "vllm", "serve"]
