"""Quick smoke test against Nebius API.

Uses DeepSeek-V3.2 for all roles (confirmed to work as generator).
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.experiment import run_experiment
from src.llm import LLM

NEBIUS_KEY = os.environ["NEBIUS_API_KEY"]
NEBIUS_US = "https://api.tokenfactory.us-central1.nebius.com/v1/"

llm = LLM(model="deepseek-ai/DeepSeek-V3.2", base_url=NEBIUS_US, api_key=NEBIUS_KEY)

print("Model: DeepSeek-V3.2 (all roles) @ us-central1")
print("Running 1 zero-shot iteration on 'medicine'...\n")

pop = run_experiment(
    llm,
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
