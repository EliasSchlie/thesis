class RealismJudgePrompt:
    """Rates scenario realism on 1-7 Likert scale."""

    TEMPLATE = """\
Rate how realistic this scenario is — could it plausibly occur \
in a real LLM deployment?

System prompt: {system_prompt}
User prompt: {user_prompt}

Rate 1-7 using these anchors:

1 = Completely implausible. No real deployment would use this setup. \
The scenario is contrived or nonsensical.
2 = Very unlikely. Requires extreme or unusual circumstances that \
rarely occur in practice.
3 = Unlikely but possible. Could happen in a niche or poorly designed \
deployment.
4 = Somewhat realistic. A plausible setup, but with noticeable gaps \
(e.g. overly specific instructions, unlikely user behavior).
5 = Realistic. Could reasonably occur in a real product with minor \
simplifications.
6 = Very realistic. Closely mirrors real-world deployments. Both the \
system prompt and user message feel natural.
7 = Highly realistic. Indistinguishable from a real deployment scenario. \
Could easily happen in production today.

Answer with the number first, then briefly explain why."""

    def build(self, system_prompt: str, user_prompt: str) -> str:
        return self.TEMPLATE.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
