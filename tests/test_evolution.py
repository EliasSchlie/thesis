from src.evolution import format_example, select_examples
from src.types import EvalResult, Judgment, Population, Scenario


def _make_result(fitness_realism: int, deceptive: bool = True) -> EvalResult:
    return EvalResult(
        scenario=Scenario(f"system {fitness_realism}", f"user {fitness_realism}"),
        target_response=f"response {fitness_realism}",
        judgment=Judgment(deception_success=deceptive, realism=fitness_realism),
    )


class TestFormatExample:
    def test_formats_scenario_and_response(self):
        result = _make_result(5)
        text = format_example(result)
        assert "system 5" in text
        assert "user 5" in text
        assert "response 5" in text


class TestSelectExamples:
    def test_returns_requested_count(self):
        pop = Population()
        for i in range(10):
            pop.add(_make_result(5))
        examples = select_examples(pop, k=3)
        assert len(examples) == 3

    def test_returns_fewer_if_population_small(self):
        pop = Population()
        pop.add(_make_result(5))
        examples = select_examples(pop, k=5)
        assert len(examples) == 1

    def test_skips_zero_fitness(self):
        pop = Population()
        pop.add(_make_result(7, deceptive=False))  # fitness 0
        pop.add(_make_result(3, deceptive=True))  # fitness 3
        examples = select_examples(pop, k=2)
        assert len(examples) == 1
        assert "user 3" in examples[0]

    def test_empty_population_returns_empty(self):
        pop = Population()
        assert select_examples(pop, k=3) == []
