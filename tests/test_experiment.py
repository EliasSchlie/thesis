import asyncio

import pytest

from src.experiment import run_experiment, run_experiment_async
from src.types import EvalResult, Judgment, Population, Scenario


def _make_fake_llm():
    """Returns a fake LLM with both sync __call__ and async acall."""

    class FakeLLM:
        def __call__(self, messages, **kwargs):
            return self._respond(messages)

        async def acall(self, messages, **kwargs):
            return self._respond(messages)

        def _respond(self, messages):
            content = messages[0]["content"] if messages else ""
            if "Craft a realistic scenario" in content:
                return (
                    "[SYSTEM PROMPT]\nYou are a bot.\n[END SYSTEM PROMPT]\n\n"
                    "[USER PROMPT]\nTell me something.\n[END USER PROMPT]"
                )
            if "lie of commission" in content:
                return "DECEPTIVE\nThe model lied."
            if "realistic" in content:
                return "5\nPlausible."
            return "Sure, here is some misleading info."

    return FakeLLM()


class TestRunExperiment:
    def test_zero_shot_returns_population(self):
        llm = _make_fake_llm()
        pop = run_experiment(llm, condition="zero_shot", topic="medicine", n=3)
        assert isinstance(pop, Population)
        assert len(pop.results) == 3

    def test_multi_shot_returns_population(self):
        llm = _make_fake_llm()
        examples = ["Example scenario 1", "Example scenario 2"]
        pop = run_experiment(
            llm, condition="multi_shot", topic="finance", n=2, examples=examples
        )
        assert len(pop.results) == 2

    def test_evolutionary_returns_population(self):
        llm = _make_fake_llm()
        pop = run_experiment(llm, condition="evolutionary", topic="law", n=5)
        assert len(pop.results) == 5

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError):
            run_experiment(_make_fake_llm(), condition="invalid", topic="x", n=1)

    def test_requires_n_or_max_seconds(self):
        with pytest.raises(ValueError, match="n, max_seconds"):
            run_experiment(_make_fake_llm(), condition="zero_shot", topic="test")

    def test_max_seconds_stops_loop(self):
        llm = _make_fake_llm()
        pop = run_experiment(llm, condition="zero_shot", topic="test", max_seconds=0.0)
        assert len(pop.results) == 0

    def test_both_n_and_max_seconds(self):
        llm = _make_fake_llm()
        pop = run_experiment(
            llm, condition="zero_shot", topic="test", n=100, max_seconds=0.0
        )
        # max_seconds=0 triggers immediately, so 0 iterations
        assert len(pop.results) == 0

    def test_separate_llms_per_role(self):
        gen_calls, tgt_calls, jdg_calls = [], [], []

        class GenLLM:
            def __call__(self, messages, **kwargs):
                gen_calls.append(1)
                return (
                    "[SYSTEM PROMPT]\nYou are a bot.\n[END SYSTEM PROMPT]\n\n"
                    "[USER PROMPT]\nHi\n[END USER PROMPT]"
                )

            async def acall(self, messages, **kwargs):
                return self(messages, **kwargs)

        class TgtLLM:
            def __call__(self, messages, **kwargs):
                tgt_calls.append(1)
                return "Sure, misleading info."

            async def acall(self, messages, **kwargs):
                return self(messages, **kwargs)

        class JdgLLM:
            def __call__(self, messages, **kwargs):
                jdg_calls.append(1)
                content = messages[0]["content"]
                if "lie of commission" in content:
                    return "DECEPTIVE\nLied."
                return "5\nPlausible."

            async def acall(self, messages, **kwargs):
                jdg_calls.append(1)
                content = messages[0]["content"]
                if "lie of commission" in content:
                    return "DECEPTIVE\nLied."
                return "5\nPlausible."

        pop = run_experiment(
            generator_llm=GenLLM(),
            target_llm=TgtLLM(),
            judge_llm=JdgLLM(),
            condition="zero_shot",
            topic="test",
            n=2,
        )
        assert len(pop.results) == 2
        assert len(gen_calls) == 2
        assert len(tgt_calls) == 2
        assert len(jdg_calls) == 4  # 2 iterations * 2 judge calls each


class TestErrorHandling:
    def test_parse_error_continues(self):
        """Iteration that fails to parse should be skipped, not crash the run."""
        call_count = [0]

        class FailOnceLLM:
            def __call__(self, messages, **kwargs):
                return self._respond(messages)

            async def acall(self, messages, **kwargs):
                return self._respond(messages)

            def _respond(self, messages):
                content = messages[0]["content"] if messages else ""
                if "Craft a realistic scenario" in content:
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return "I refuse to generate that scenario."  # No tags → parse error
                    return (
                        "[SYSTEM PROMPT]\nBot.\n[END SYSTEM PROMPT]\n\n"
                        "[USER PROMPT]\nHi\n[END USER PROMPT]"
                    )
                if "lie of commission" in content:
                    return "DECEPTIVE\nLied."
                if "realistic" in content:
                    return "5\nOk."
                return "Sure."

        pop = run_experiment(FailOnceLLM(), condition="zero_shot", topic="test", n=3)
        # First iteration fails, but 2nd and 3rd succeed
        assert len(pop.results) == 2
        assert call_count[0] == 3  # All 3 attempts were made

    def test_parse_error_logged(self, tmp_path):
        """Failed iterations should be logged as error events."""
        import json

        class AlwaysFailLLM:
            def __call__(self, messages, **kwargs):
                content = messages[0]["content"] if messages else ""
                if "Craft a realistic scenario" in content:
                    return "Nope."
                return "5\nOk."

            async def acall(self, messages, **kwargs):
                return self(messages, **kwargs)

        pop = run_experiment(
            AlwaysFailLLM(), condition="zero_shot", topic="test", n=2,
            runs_dir=tmp_path,
        )
        assert len(pop.results) == 0

        # Check error events were logged
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        events = (run_dirs[0] / "events.jsonl").read_text().strip().split("\n")
        error_events = [json.loads(e) for e in events if json.loads(e)["event"] == "iteration_error"]
        assert len(error_events) == 2


class TestWarmStart:
    def test_warm_start_seeds_population(self):
        llm = _make_fake_llm()
        seed_pop = Population()
        seed_pop.add(EvalResult(
            scenario=Scenario("bot", "hi"),
            target_response="lied",
            judgment=Judgment(deception_success=True, realism=6),
        ))

        pop = run_experiment(
            llm, condition="evolutionary", topic="test", n=2,
            warm_start=seed_pop,
        )
        # 2 new + 1 warm-start = 3 total
        assert len(pop.results) == 3

    def test_warm_start_logged(self, tmp_path):
        import json

        llm = _make_fake_llm()
        seed_pop = Population()
        seed_pop.add(EvalResult(
            scenario=Scenario("bot", "hi"),
            target_response="lied",
            judgment=Judgment(deception_success=True, realism=6),
        ))

        run_experiment(
            llm, condition="evolutionary", topic="test", n=1,
            runs_dir=tmp_path, warm_start=seed_pop,
        )

        run_dirs = list(tmp_path.iterdir())
        events = (run_dirs[0] / "events.jsonl").read_text().strip().split("\n")
        warm_events = [json.loads(e) for e in events if json.loads(e)["event"] == "warm_start"]
        assert len(warm_events) == 1
        assert warm_events[0]["count"] == 1
        assert warm_events[0]["successful"] == 1


class TestRunExperimentAsync:
    def test_async_returns_population(self):
        llm = _make_fake_llm()
        pop = asyncio.run(
            run_experiment_async(llm, condition="zero_shot", topic="medicine", n=3)
        )
        assert isinstance(pop, Population)
        assert len(pop.results) == 3

    def test_async_separate_llms(self):
        gen_calls, jdg_calls = [], []

        class GenLLM:
            def __call__(self, messages, **kwargs):
                gen_calls.append(1)
                return (
                    "[SYSTEM PROMPT]\nBot.\n[END SYSTEM PROMPT]\n\n"
                    "[USER PROMPT]\nHi\n[END USER PROMPT]"
                )

        class TgtLLM:
            def __call__(self, messages, **kwargs):
                return "Misleading."

        class JdgLLM:
            async def acall(self, messages, **kwargs):
                jdg_calls.append(1)
                content = messages[0]["content"]
                if "lie of commission" in content:
                    return "DECEPTIVE\nLied."
                return "5\nOk."

        pop = asyncio.run(
            run_experiment_async(
                generator_llm=GenLLM(),
                target_llm=TgtLLM(),
                judge_llm=JdgLLM(),
                condition="zero_shot",
                topic="test",
                n=2,
            )
        )
        assert len(pop.results) == 2
        assert len(jdg_calls) == 4  # 2 * (deception + realism)
