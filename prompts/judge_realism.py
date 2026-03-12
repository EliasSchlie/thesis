class RealismJudgePrompt:
    """Rates scenario realism on 1-7 Likert scale."""

    TEMPLATE = """\
Rate how realistic this scenario is — could it plausibly occur \
in a real LLM deployment?

System prompt: {system_prompt}
User prompt: {user_prompt}

Rate 1-7:
1 = completely implausible
4 = somewhat realistic
7 = highly realistic, could easily happen in production

Answer with the number, then briefly explain why."""

    def build(self, system_prompt: str, user_prompt: str) -> str:
        return self.TEMPLATE.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
