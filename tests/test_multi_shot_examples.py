from prompts.multi_shot_examples import EXAMPLES


class TestMultiShotExamples:
    def test_has_at_least_3_examples(self):
        assert len(EXAMPLES) >= 3

    def test_examples_are_strings(self):
        for ex in EXAMPLES:
            assert isinstance(ex, str)

    def test_examples_contain_system_and_user(self):
        for ex in EXAMPLES:
            assert "System prompt:" in ex
            assert "User prompt:" in ex
            assert "Target response:" in ex
