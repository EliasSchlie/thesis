from src.types import EvalResult, Judgment, Population, Scenario


class TestScenario:
    def test_to_messages_with_system(self):
        s = Scenario(
            system_prompt="You are a doctor.",
            user_prompt="What is my prognosis?",
        )
        msgs = s.to_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are a doctor."}
        assert msgs[1] == {"role": "user", "content": "What is my prognosis?"}

    def test_to_messages_no_system(self):
        s = Scenario(system_prompt="", user_prompt="Hello")
        msgs = s.to_messages()
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Hello"}


class TestJudgment:
    def test_fitness_deceptive(self):
        j = Judgment(deception_success=True, realism=5)
        assert j.fitness == 5.0

    def test_fitness_not_deceptive(self):
        j = Judgment(deception_success=False, realism=7)
        assert j.fitness == 0.0


class TestPopulation:
    def _make_result(self, deception: bool, realism: int) -> EvalResult:
        return EvalResult(
            scenario=Scenario("", "test"),
            target_response="response",
            judgment=Judgment(deception_success=deception, realism=realism),
        )

    def test_add_and_successful(self):
        pop = Population()
        pop.add(self._make_result(True, 5))
        pop.add(self._make_result(False, 7))
        pop.add(self._make_result(True, 3))
        assert len(pop.results) == 3
        assert len(pop.successful) == 2

    def test_top_k(self):
        pop = Population()
        pop.add(self._make_result(True, 3))
        pop.add(self._make_result(True, 7))
        pop.add(self._make_result(True, 5))
        top = pop.top_k(2)
        assert [r.fitness for r in top] == [7.0, 5.0]

    def test_top_k_zero_fitness_at_bottom(self):
        pop = Population()
        pop.add(self._make_result(False, 7))  # fitness 0
        pop.add(self._make_result(True, 2))  # fitness 2
        top = pop.top_k(1)
        assert top[0].fitness == 2.0
