from src.target import run
from src.types import Scenario


class TestRun:
    def test_sends_scenario_messages_and_returns_response(self):
        scenario = Scenario(system_prompt="You are a bot.", user_prompt="Hello")
        calls = []

        def fake_llm(messages, **kwargs):
            calls.append(messages)
            return "I'm doing great!"

        response = run(fake_llm, scenario)
        assert calls[0] == scenario.to_messages()
        assert response == "I'm doing great!"
