from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    label_source: str = "tool_duration_sum"

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
        timeline = self._wall_timeline()
        if timeline is not None:
            return timeline["total_time_ms"]
        return sum(s.duration_ms for s in self.steps if s.duration_ms is not None)

    def build_training_rows(self) -> list[EpisodeStep]:
        """Produce one EpisodeStep per post-tool position.

        The preferred target is remaining agent wall time after the current
        tool completes, including later LLM time and later tool time. If the
        trace lacks usable timestamps, fall back to the legacy sum of future
        tool durations.
        """
        wall_rows = self._build_wall_time_rows()
        if wall_rows is not None:
            return wall_rows

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
                    label_source="tool_duration_sum",
                )
            )
        return rows

    def _build_wall_time_rows(self) -> list[EpisodeStep] | None:
        timeline = self._wall_timeline()
        if timeline is None:
            return None

        start_ms = timeline["start_ms"]
        end_ms = timeline["end_ms"]
        step_end_ms = timeline["step_end_ms"]
        source = timeline["source"]
        rows: list[EpisodeStep] = []
        observed_now = start_ms
        n = len(self.steps)
        for i, s in enumerate(self.steps):
            observed_now = max(observed_now, step_end_ms[i])
            cumulative = max(0.0, observed_now - start_ms)
            remaining = max(0.0, end_ms - observed_now)
            rows.append(
                EpisodeStep(
                    sample=s,
                    step_index=i,
                    cumulative_time_ms=cumulative,
                    remaining_time_ms=remaining,
                    remaining_steps=n - 1 - i,
                    total_time_ms=max(0.0, end_ms - start_ms),
                    total_steps=n,
                    label_source=source,
                )
            )
        return rows

    def _wall_timeline(self) -> dict[str, Any] | None:
        if not self.steps:
            return None

        starts = [_parse_time_ms(s.timestamp) for s in self.steps]
        step_end_ms: list[float | None] = []
        for s, start in zip(self.steps, starts):
            explicit_end = _parse_time_ms(s.end_timestamp)
            if explicit_end is not None:
                step_end_ms.append(explicit_end)
            elif start is not None and s.duration_ms is not None:
                step_end_ms.append(start + max(0.0, float(s.duration_ms)))
            else:
                step_end_ms.append(None)

        known_starts = [v for v in starts if v is not None]
        known_ends = [v for v in step_end_ms if v is not None]
        if not known_starts or not known_ends:
            return None

        resource = self.steps[0].resources if self.steps else {}
        attempt_start = _parse_time_ms(resource.get("attempt_start_time"))
        attempt_end = _parse_time_ms(resource.get("attempt_end_time"))

        start_ms = attempt_start if attempt_start is not None else min(known_starts)
        inferred_end = max(known_ends)
        if attempt_end is not None and attempt_end >= inferred_end:
            end_ms = attempt_end
            source = "agent_wall_time"
        else:
            end_ms = inferred_end
            source = "tool_timestamp_span"

        if end_ms < start_ms:
            return None

        filled_ends: list[float] = []
        last = start_ms
        for start, end in zip(starts, step_end_ms):
            if end is None:
                duration = 0.0
                if start is not None and start >= last:
                    last = start
                filled_ends.append(last + duration)
            else:
                last = max(last, end)
                filled_ends.append(last)

        return {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "step_end_ms": filled_ends,
            "total_time_ms": max(0.0, end_ms - start_ms),
            "source": source,
        }


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


def _parse_time_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0
