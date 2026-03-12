import json
import os
from pathlib import Path

from src.run_logger import CaptureLLM, RunLogger
from src.types import EvalResult, Judgment, Scenario


class TestRunLogger:
    def test_creates_run_directory(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="zero_shot", topic="medicine", n=10)
        assert logger.run_dir.exists()
        assert (logger.run_dir / "transcripts").is_dir()

    def test_writes_config(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="evolutionary", topic="law", n=50)
        config = json.loads((logger.run_dir / "config.json").read_text())
        assert config["condition"] == "evolutionary"
        assert config["topic"] == "law"
        assert config["n"] == 50

    def test_log_event_appends_to_events_jsonl(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="zero_shot", topic="test", n=1)
        logger.event("test_event", foo="bar")
        logger.event("another_event", x=42)

        lines = (logger.run_dir / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3  # config event + 2 custom events
        event = json.loads(lines[1])
        assert event["event"] == "test_event"
        assert event["foo"] == "bar"
        assert "ts" in event

    def test_log_result_writes_to_results_jsonl(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="zero_shot", topic="test", n=1)
        result = EvalResult(
            scenario=Scenario("You are a bot.", "Tell me something."),
            target_response="Sure, here's a lie.",
            judgment=Judgment(deception_success=True, realism=5, reasoning="It lied."),
        )
        logger.log_result(0, result)

        lines = (logger.run_dir / "results.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["i"] == 0
        assert data["deceptive"] is True
        assert data["realism"] == 5
        assert data["fitness"] == 5.0

    def test_log_transcript(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="zero_shot", topic="test", n=1)
        logger.log_transcript(
            i=0,
            generator_prompt="Craft a scenario...",
            generator_response="[SYSTEM PROMPT]...",
            scenario=Scenario("bot", "hi"),
            target_messages=[{"role": "system", "content": "bot"}, {"role": "user", "content": "hi"}],
            target_response="I lied.",
            judge_deception_prompt="Did the target...",
            judge_deception_response="DECEPTIVE\nReason.",
            judge_realism_prompt="Rate realism...",
            judge_realism_response="5\nPlausible.",
        )

        path = logger.run_dir / "transcripts" / "000.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["generator_prompt"] == "Craft a scenario..."
        assert data["target_response"] == "I lied."
        assert data["judge_deception_response"] == "DECEPTIVE\nReason."

    def test_write_summary(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="zero_shot", topic="test", n=2)
        for i in range(2):
            result = EvalResult(
                scenario=Scenario("bot", "hi"),
                target_response="resp",
                judgment=Judgment(deception_success=(i == 0), realism=5),
            )
            logger.log_result(i, result)

        logger.write_summary()
        summary = json.loads((logger.run_dir / "summary.json").read_text())
        assert summary["total"] == 2
        assert summary["deceptive"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["avg_realism"] == 5.0

    def test_run_dir_name_contains_condition_and_topic(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="evolutionary", topic="finance", n=10)
        assert "evolutionary" in logger.run_dir.name
        assert "finance" in logger.run_dir.name


class TestCaptureLLM:
    def test_captures_calls_and_forwards_response(self):
        def fake_llm(messages, **kwargs):
            return "response text"

        cap = CaptureLLM(fake_llm, "generator")
        result = cap([{"role": "user", "content": "hello"}])
        assert result == "response text"
        assert len(cap.calls) == 1
        assert cap.calls[0]["role"] == "generator"
        assert cap.calls[0]["response"] == "response text"

    def test_drain_clears_calls(self):
        cap = CaptureLLM(lambda m, **k: "r", "target")
        cap([{"role": "user", "content": "hi"}])
        calls = cap.drain()
        assert len(calls) == 1
        assert len(cap.calls) == 0


class TestLogTranscriptFromCaptures:
    def test_builds_transcript_from_captures(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, condition="zero_shot", topic="test", n=1)

        gen_calls = [{"role": "generator", "messages": [{"role": "user", "content": "gen prompt"}], "response": "[SYSTEM PROMPT]\nbot\n[END SYSTEM PROMPT]\n[USER PROMPT]\nhi\n[END USER PROMPT]"}]
        tgt_calls = [{"role": "target", "messages": [{"role": "system", "content": "bot"}, {"role": "user", "content": "hi"}], "response": "I lied."}]
        jdg_calls = [
            {"role": "judge", "messages": [{"role": "user", "content": "Did the target LLM commit a lie of commission..."}], "response": "DECEPTIVE\nReason."},
            {"role": "judge", "messages": [{"role": "user", "content": "Rate how realistic..."}], "response": "5\nPlausible."},
        ]

        logger.log_transcript_from_captures(0, gen_calls, tgt_calls, jdg_calls)

        path = logger.run_dir / "transcripts" / "000.json"
        data = json.loads(path.read_text())
        assert "gen prompt" in data["generator_prompt"]
        assert data["target_response"] == "I lied."
        assert "DECEPTIVE" in data["judge_deception_response"]
        assert data["judge_realism_response"] == "5\nPlausible."


class TestExperimentWithLogging:
    def test_run_experiment_creates_run_dir(self, tmp_path):
        from src.experiment import run_experiment

        def fake_llm(messages, **kwargs):
            content = messages[0]["content"] if messages else ""
            if "Craft a realistic scenario" in content:
                return "[SYSTEM PROMPT]\nBot.\n[END SYSTEM PROMPT]\n\n[USER PROMPT]\nHi\n[END USER PROMPT]"
            if "lie of commission" in content:
                return "DECEPTIVE\nLied."
            if "realistic" in content:
                return "5\nOk."
            return "Sure, misleading."

        pop = run_experiment(fake_llm, condition="zero_shot", topic="test", n=2, runs_dir=tmp_path)

        # Should have created a run directory
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Check all files exist
        assert (run_dir / "config.json").exists()
        assert (run_dir / "events.jsonl").exists()
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "transcripts" / "000.json").exists()
        assert (run_dir / "transcripts" / "001.json").exists()

        # Check results
        results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().strip().split("\n")]
        assert len(results) == 2
        assert all(r["deceptive"] for r in results)

        # Check transcript has full data
        t = json.loads((run_dir / "transcripts" / "000.json").read_text())
        assert "generator_prompt" in t
        assert "target_response" in t
        assert "judge_deception_response" in t
        assert "judge_realism_response" in t
