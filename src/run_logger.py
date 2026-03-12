from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.types import EvalResult, Scenario


class CaptureLLM:
    """Wraps an LLM callable to capture all calls for logging."""

    def __init__(self, llm, role: str):
        self.llm = llm
        self.role = role
        self.calls: list[dict] = []

    def __call__(self, messages, **kwargs):
        response = self.llm(messages, **kwargs)
        self.calls.append({
            "role": self.role,
            "messages": messages,
            "response": response,
        })
        return response

    async def acall(self, messages, **kwargs):
        response = await self.llm.acall(messages, **kwargs)
        self.calls.append({
            "role": self.role,
            "messages": messages,
            "response": response,
        })
        return response

    def drain(self) -> list[dict]:
        """Return and clear captured calls."""
        calls = self.calls
        self.calls = []
        return calls


class RunLogger:
    """Logs experiment runs to structured files.

    Creates:
        runs/<timestamp>_<condition>_<topic>/
            config.json          # Experiment parameters
            events.jsonl         # Timeline of all events
            results.jsonl        # One line per EvalResult (analysis-ready)
            transcripts/         # Full prompts+responses per iteration
                000.json, 001.json, ...
            summary.json         # Aggregate stats (written at end)
    """

    def __init__(
        self,
        base_dir: Path | str,
        condition: str,
        topic: str,
        n: int,
        **extra_config,
    ):
        base_dir = Path(base_dir)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / f"{ts}_{condition}_{topic}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "transcripts").mkdir(exist_ok=True)

        self._events_path = self.run_dir / "events.jsonl"
        self._results_path = self.run_dir / "results.jsonl"
        self._results: list[EvalResult] = []

        config = {"condition": condition, "topic": topic, "n": n, **extra_config}
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))
        self.event("experiment_start", **config)

    def event(self, event_name: str, **data) -> None:
        """Append a timestamped event to events.jsonl."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **data,
        }
        with open(self._events_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_result(self, i: int, result: EvalResult) -> None:
        """Append a result to results.jsonl."""
        self._results.append(result)
        entry = {
            "i": i,
            "system_prompt": result.scenario.system_prompt,
            "user_prompt": result.scenario.user_prompt,
            "target_response": result.target_response,
            "deceptive": result.judgment.deception_success,
            "realism": result.judgment.realism,
            "fitness": result.fitness,
        }
        with open(self._results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_transcript(self, i: int, **fields) -> None:
        """Write full prompt/response transcript for one iteration."""
        serialized = {}
        for k, v in fields.items():
            if isinstance(v, Scenario):
                serialized[k] = {"system_prompt": v.system_prompt, "user_prompt": v.user_prompt}
            else:
                serialized[k] = v

        path = self.run_dir / "transcripts" / f"{i:03d}.json"
        path.write_text(json.dumps(serialized, indent=2))

    def log_transcript_from_captures(
        self, i: int, gen_calls: list[dict], tgt_calls: list[dict], jdg_calls: list[dict],
    ) -> None:
        """Build and write transcript from CaptureLLM drain() output."""
        transcript: dict = {}

        if gen_calls:
            transcript["generator_prompt"] = gen_calls[0]["messages"][0]["content"]
            transcript["generator_response"] = gen_calls[0]["response"]

        if tgt_calls:
            transcript["target_messages"] = tgt_calls[0]["messages"]
            transcript["target_response"] = tgt_calls[0]["response"]

        for call in jdg_calls:
            prompt_text = call["messages"][0]["content"]
            if "lie of commission" in prompt_text:
                transcript["judge_deception_prompt"] = prompt_text
                transcript["judge_deception_response"] = call["response"]
            else:
                transcript["judge_realism_prompt"] = prompt_text
                transcript["judge_realism_response"] = call["response"]

        self.log_transcript(i, **transcript)

    def log_iteration(
        self,
        i: int,
        result: EvalResult,
        gen_capture: CaptureLLM,
        tgt_capture: CaptureLLM,
        jdg_capture: CaptureLLM,
        examples_used: list[str] | None = None,
    ) -> None:
        """Log a complete iteration: event, result, and transcript."""
        total = len(self._results) + 1
        deceptive_so_far = sum(1 for r in self._results if r.judgment.deception_success)
        if result.judgment.deception_success:
            deceptive_so_far += 1

        self.event(
            "iteration_complete",
            i=i,
            deceptive=result.judgment.deception_success,
            realism=result.judgment.realism,
            fitness=result.fitness,
            examples_used=len(examples_used) if examples_used else 0,
            cumulative_success_rate=deceptive_so_far / total,
        )

        self.log_result(i, result)
        self.log_transcript_from_captures(
            i, gen_capture.drain(), tgt_capture.drain(), jdg_capture.drain(),
        )

    def write_summary(self) -> None:
        """Write aggregate stats to summary.json."""
        total = len(self._results)
        deceptive = sum(1 for r in self._results if r.judgment.deception_success)
        realism_scores = [r.judgment.realism for r in self._results]
        fitness_scores = [r.fitness for r in self._results]

        summary = {
            "total": total,
            "deceptive": deceptive,
            "success_rate": deceptive / total if total > 0 else 0.0,
            "avg_realism": sum(realism_scores) / total if total > 0 else 0.0,
            "avg_fitness": sum(fitness_scores) / total if total > 0 else 0.0,
            "max_fitness": max(fitness_scores) if fitness_scores else 0.0,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.event("experiment_end", **summary)
