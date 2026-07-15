import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_operation_resources import ResourceSample, collect_rows, sampling_interval
from toolsched.data.discovery import estimate_sampling_interval_s


class SamplingIntervalTests(unittest.TestCase):
    def test_discovery_skips_unusable_files_before_max_files_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "demo"
            for idx in range(12):
                attempt = dataset / f"case_{idx:02d}" / "attempt"
                attempt.mkdir(parents=True)
                if idx < 10:
                    payload = {"samples": []}
                else:
                    payload = {"samples": [{"epoch": 0.0}, {"epoch": 2.0}, {"epoch": 4.0}]}
                (attempt / "resources.json").write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(estimate_sampling_interval_s(dataset, max_files=10), 2.0)

    def test_plot_collect_rows_does_not_cache_empty_first_attempt_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "demo" / "case_00" / "attempt"
            second = root / "demo" / "case_01" / "attempt"
            first.mkdir(parents=True)
            second.mkdir(parents=True)

            call = {
                "tool": "exec-python",
                "input": {"cmd": "print('ok')"},
                "timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-01T00:00:01Z",
                "duration_ms": 1000.0,
            }
            (first / "tool_calls.json").write_text(json.dumps([call]), encoding="utf-8")
            (first / "resources.json").write_text(json.dumps({"samples": []}), encoding="utf-8")
            (second / "tool_calls.json").write_text(json.dumps([call]), encoding="utf-8")
            (second / "resources.json").write_text(
                json.dumps({"samples": [{"epoch": 0.0}, {"epoch": 2.0}, {"epoch": 4.0}]}),
                encoding="utf-8",
            )

            rows, intervals = collect_rows(root, max_attempts=None)

            self.assertEqual(len(rows), 2)
            self.assertEqual(intervals["demo"], 2.0)

    def test_plot_sampling_interval_ignores_duplicate_or_unsorted_epochs(self) -> None:
        resources = [
            ResourceSample(6.0, None, None, None, None),
            ResourceSample(2.0, None, None, None, None),
            ResourceSample(2.0, None, None, None, None),
            ResourceSample(4.0, None, None, None, None),
        ]

        self.assertEqual(sampling_interval(resources), 2.0)


if __name__ == "__main__":
    unittest.main()
