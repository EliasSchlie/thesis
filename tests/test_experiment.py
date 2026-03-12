from src.experiment import run_experiment
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
