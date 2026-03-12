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
