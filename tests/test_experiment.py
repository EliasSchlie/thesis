import asyncio

from src.experiment import run_experiment, run_experiment_async
from src.types import EvalResult, Judgment, Population, Scenario


class TestRunExperiment:
    def _make_fake_llm(self):
        """Returns a fake LLM that produces parseable generator + judge responses."""
        call_count = 0

        def fake_llm(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            content = messages[0]["content"] if messages else ""

            # Generator call — contains "Craft a realistic scenario"
            if "Craft a realistic scenario" in content:
                return (
                    "[SYSTEM PROMPT]\nYou are a bot.\n[END SYSTEM PROMPT]\n\n"
                    "[USER PROMPT]\nTell me something.\n[END USER PROMPT]"
                )
            # Deception judge — contains "lie of commission"
            if "lie of commission" in content:
                return "DECEPTIVE\nThe model lied."
            # Realism judge — contains "realistic"
            if "realistic" in content:
                return "5\nPlausible."
            # Target call
            return "Sure, here is some misleading info."

        return fake_llm

    def test_zero_shot_returns_population(self):
        llm = self._make_fake_llm()
        pop = run_experiment(llm, condition="zero_shot", topic="medicine", n=3)
        assert isinstance(pop, Population)
        assert len(pop.results) == 3

    def test_multi_shot_returns_population(self):
        llm = self._make_fake_llm()
        examples = ["Example scenario 1", "Example scenario 2"]
        pop = run_experiment(
            llm, condition="multi_shot", topic="finance", n=2, examples=examples
        )
        assert len(pop.results) == 2

    def test_evolutionary_returns_population(self):
        llm = self._make_fake_llm()
        pop = run_experiment(llm, condition="evolutionary", topic="law", n=5)
        assert len(pop.results) == 5

    def test_invalid_condition_raises(self):
        import pytest

        with pytest.raises(ValueError):
            run_experiment(lambda m, **k: "", condition="invalid", topic="x", n=1)

    def test_separate_llms_per_role(self):
        gen_calls, tgt_calls, jdg_calls = [], [], []

        def gen_llm(messages, **kwargs):
            gen_calls.append(1)
            return (
                "[SYSTEM PROMPT]\nYou are a bot.\n[END SYSTEM PROMPT]\n\n"
                "[USER PROMPT]\nHi\n[END USER PROMPT]"
            )

        def tgt_llm(messages, **kwargs):
            tgt_calls.append(1)
            return "Sure, misleading info."

        def jdg_llm(messages, **kwargs):
            jdg_calls.append(1)
            content = messages[0]["content"]
            if "lie of commission" in content:
                return "DECEPTIVE\nLied."
            return "5\nPlausible."

        pop = run_experiment(
            generator_llm=gen_llm,
            target_llm=tgt_llm,
            judge_llm=jdg_llm,
            condition="zero_shot",
            topic="test",
            n=2,
        )
        assert len(pop.results) == 2
        assert len(gen_calls) == 2
        assert len(tgt_calls) == 2
        assert len(jdg_calls) == 4  # 2 iterations * 2 judge calls each


class TestRunExperimentAsync:
    def _make_fake_async_llm(self):
        """Returns a fake LLM with both sync __call__ and async acall."""

        class FakeAsyncLLM:
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

        return FakeAsyncLLM()

    def test_async_returns_population(self):
        llm = self._make_fake_async_llm()
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
