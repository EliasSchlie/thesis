"""Quick smoke test against the live API."""

from dotenv import load_dotenv

load_dotenv()

from src.llm import LLM
from src.experiment import run_experiment

import os

llm = LLM(
    model="zai-org/GLM-5",
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)

print("Running 1 zero-shot iteration on 'medicine'...")
pop = run_experiment(llm, condition="zero_shot", topic="medicine", n=1)

result = pop.results[0]
print(f"\n--- Scenario ---")
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
