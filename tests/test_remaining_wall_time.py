import unittest

from toolsched.episodes import AgentEpisode
from toolsched.schema import ToolSample


def sample(idx: int, start: str, end: str, duration_ms: float) -> ToolSample:
    return ToolSample(
        sample_id=f"case/tool_{idx}",
        dataset="demo",
        case_id="case",
        attempt_id="attempt",
        tool="exec-python",
        operation="data_script",
        tool_family="data_analysis_scripting",
        timestamp=start,
        duration_ms=duration_ms,
        end_timestamp=end,
        features={"call_index": idx},
    )


class RemainingWallTimeTests(unittest.TestCase):
    def test_remaining_time_uses_agent_wall_clock(self) -> None:
        first = sample(0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1000.0)
        second = sample(1, "2026-01-01T00:00:05Z", "2026-01-01T00:00:06Z", 1000.0)
        for row in (first, second):
            row.resources["attempt_start_time"] = "2026-01-01T00:00:00Z"
            row.resources["attempt_end_time"] = "2026-01-01T00:00:10Z"

        rows = AgentEpisode("demo", "case", "attempt", [first, second]).build_training_rows()

        self.assertEqual(rows[0].label_source, "agent_wall_time")
        self.assertEqual(rows[0].cumulative_time_ms, 1000.0)
        self.assertEqual(rows[0].remaining_time_ms, 9000.0)
        self.assertEqual(rows[1].remaining_time_ms, 4000.0)


if __name__ == "__main__":
    unittest.main()
