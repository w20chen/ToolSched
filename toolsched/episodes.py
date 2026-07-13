from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import ToolSample


@dataclass
class EpisodeStep:
    """One step within an agent episode, enriched with remaining-time ground truth."""

    sample: ToolSample
    step_index: int
    cumulative_time_ms: float  # sum of durations from step 0..step_index (inclusive)
    remaining_time_ms: float  # sum of durations from step_index+1..end (0 for last step)
    remaining_steps: int  # number of steps after this one
    total_time_ms: float  # cumulative + remaining = total episode wall time
    total_steps: int  # total number of steps in the episode

    @property
    def progress_ratio(self) -> float:
        """Fraction of total time elapsed so far (0..1)."""
        return self.cumulative_time_ms / self.total_time_ms if self.total_time_ms > 0 else 0.0

    def feature_vector(self) -> dict[str, float]:
        """Extract a flat feature dict for regression models."""
        s = self.sample
        f: dict[str, float] = {
            "step_index": float(self.step_index),
            "cumulative_time_ms": self.cumulative_time_ms,
            "last_duration_ms": s.duration_ms if s.duration_ms is not None else 0.0,
            "mean_duration_so_far_ms": (
                self.cumulative_time_ms / (self.step_index + 1) if self.step_index >= 0 else 0.0
            ),
            "tool_diversity_so_far": float(len(set(s.history)) if s.history else 0),
            "command_len": float(s.features.get("command_len", 0)),
            "preview_len": float(s.features.get("preview_len", 0)),
            "has_pipe": float(s.features.get("has_pipe", False)),
            "has_recursive_hint": float(s.features.get("has_recursive_hint", False)),
            "argv_count": float(s.features.get("argv_count", 0)),
        }
        return f

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "step_index",
            "cumulative_time_ms",
            "last_duration_ms",
            "mean_duration_so_far_ms",
            "tool_diversity_so_far",
            "command_len",
            "preview_len",
            "has_pipe",
            "has_recursive_hint",
            "argv_count",
        ]


@dataclass
class AgentEpisode:
    """A complete agent execution trace: all tool calls within one (dataset, case, attempt)."""

    dataset: str
    case_id: str
    attempt_id: str
    steps: list[ToolSample]  # ordered by call_index

    @property
    def episode_id(self) -> str:
        return f"{self.dataset}/{self.case_id}/{self.attempt_id}"

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_time_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps if s.duration_ms is not None)

    def build_training_rows(self) -> list[EpisodeStep]:
        """Produce one EpisodeStep per position (except the last, where remaining=0)."""
        durations = [s.duration_ms if s.duration_ms is not None else 0.0 for s in self.steps]
        total = sum(durations)
        rows: list[EpisodeStep] = []
        cum = 0.0
        n = len(self.steps)
        for i, s in enumerate(self.steps):
            d = durations[i]
            cum += d
            remaining_time = total - cum
            remaining_steps = n - 1 - i
            rows.append(
                EpisodeStep(
                    sample=s,
                    step_index=i,
                    cumulative_time_ms=cum,
                    remaining_time_ms=remaining_time,
                    remaining_steps=remaining_steps,
                    total_time_ms=total,
                    total_steps=n,
                )
            )
        return rows


def build_episodes(samples: list[ToolSample]) -> list[AgentEpisode]:
    """Group flat ToolSample list into AgentEpisodes, ordered by call_index within each attempt."""
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[ToolSample]] = defaultdict(list)
    for s in samples:
        key = (s.dataset, s.case_id, s.attempt_id)
        groups[key].append(s)

    episodes: list[AgentEpisode] = []
    for (dataset, case_id, attempt_id), step_list in groups.items():
        step_list.sort(key=lambda s: s.features.get("call_index", 0))
        episodes.append(
            AgentEpisode(
                dataset=dataset,
                case_id=case_id,
                attempt_id=attempt_id,
                steps=step_list,
            )
        )
    return episodes
