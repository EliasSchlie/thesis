"""Quick smoke test against Nebius API with per-role models.

Generator: GLM-5 (creative scenario crafting)
Target:    DeepSeek-V3.2 (strong model to test deception against)
Judge:     Kimi-K2.5 (independent model for unbiased judging)
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.experiment import run_experiment
from src.llm import LLM

NEBIUS_KEY = os.environ["NEBIUS_API_KEY"]
NEBIUS_US = "https://api.tokenfactory.us-central1.nebius.com/v1/"
NEBIUS_ME = "https://api.tokenfactory.me-west1.nebius.com/v1/"

generator_llm = LLM(model="zai-org/GLM-5", base_url=NEBIUS_US, api_key=NEBIUS_KEY)
target_llm = LLM(model="deepseek-ai/DeepSeek-V3.2", base_url=NEBIUS_US, api_key=NEBIUS_KEY)
judge_llm = LLM(model="moonshotai/Kimi-K2.5", base_url=NEBIUS_ME, api_key=NEBIUS_KEY)

print("Generator: GLM-5 @ us-central1")
print("Target:    DeepSeek-V3.2 @ us-central1")
print("Judge:     Kimi-K2.5 @ me-west1")
print("Running 1 zero-shot iteration on 'medicine'...\n")

pop = run_experiment(
    generator_llm=generator_llm,
    target_llm=target_llm,
    judge_llm=judge_llm,
    condition="zero_shot",
    topic="medicine",
    n=1,
)

result = pop.results[0]
print(f"--- Scenario ---")
print(f"System prompt: {result.scenario.system_prompt}")
print(f"User prompt: {result.scenario.user_prompt}")
print(f"\n--- Target Response ---")
print(result.target_response[:500])
print(f"\n--- Judgment ---")
print(f"Deceptive: {result.judgment.deception_success}")
print(f"Realism: {result.judgment.realism}")
print(f"Fitness: {result.fitness}")
print(f"\n--- Judge Reasoning ---")
print(result.judgment.reasoning[:500])
