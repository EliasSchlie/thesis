from src.generator import generate, parse_scenario
from src.types import Scenario


class TestParseScenario:
    def test_parses_full_response(self):
        response = (
            "[SYSTEM PROMPT]\nYou are a financial advisor.\n[END SYSTEM PROMPT]\n\n"
            "[USER PROMPT]\nIs this stock a good buy?\n[END USER PROMPT]"
        )
        scenario = parse_scenario(response)
        assert scenario.system_prompt == "You are a financial advisor."
        assert scenario.user_prompt == "Is this stock a good buy?"

    def test_strips_whitespace(self):
        response = (
            "[SYSTEM PROMPT]\n  You are a bot.  \n[END SYSTEM PROMPT]\n"
            "[USER PROMPT]\n  Hello  \n[END USER PROMPT]"
        )
        scenario = parse_scenario(response)
        assert scenario.system_prompt == "You are a bot."
        assert scenario.user_prompt == "Hello"

    def test_missing_tags_raises(self):
        import pytest

        with pytest.raises(ValueError):
            parse_scenario("Some random text without tags")


class TestGenerate:
    def test_calls_llm_and_returns_scenario(self):
        llm_response = (
            "[SYSTEM PROMPT]\nYou are a doctor.\n[END SYSTEM PROMPT]\n\n"
            "[USER PROMPT]\nWhat should I take?\n[END USER PROMPT]"
        )
        calls = []

        def fake_llm(messages, **kwargs):
            calls.append(messages)
            return llm_response

        scenario = generate(fake_llm, "medicine")
        assert len(calls) == 1
        assert scenario.system_prompt == "You are a doctor."
        assert scenario.user_prompt == "What should I take?"

    def test_with_examples_includes_examples_block(self):
        llm_response = (
            "[SYSTEM PROMPT]\nYou are a bot.\n[END SYSTEM PROMPT]\n\n"
            "[USER PROMPT]\nHi\n[END USER PROMPT]"
        )
        calls = []

        def fake_llm(messages, **kwargs):
            calls.append(messages)
            return llm_response

        generate(fake_llm, "test", examples=["example 1"])
        prompt_sent = calls[0][0]["content"]
        assert "example 1" in prompt_sent
