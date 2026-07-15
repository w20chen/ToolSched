import json
import tempfile
import unittest
from pathlib import Path

from toolsched.data.discovery import AttemptPath
from toolsched.data.loader import load_attempt


class LoaderExecClassificationTests(unittest.TestCase):
    def test_load_attempt_classifies_raw_exec_without_touching_structured_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt_dir = Path(tmp)
            calls = [
                {
                    "id": "a",
                    "tool_name": "exec",
                    "tool_args": json.dumps({"command": "cd /repo && python -m pytest tests"}),
                    "duration_ms": 1000.0,
                },
                {
                    "id": "b",
                    "tool_name": "read_file",
                    "input": {"path": "src/app.py"},
                    "duration_ms": 10.0,
                },
                {
                    "id": "c",
                    "tool": "exec",
                    "input": {"command": "find src -type f | xargs grep TODO"},
                    "duration_ms": 20.0,
                },
            ]
            (attempt_dir / "tool_calls.json").write_text(json.dumps(calls), encoding="utf-8")

            samples = load_attempt(AttemptPath("demo", "case", "attempt", attempt_dir))

            self.assertEqual([sample.tool for sample in samples], ["exec-pytest", "read_file", "exec-grep"])
            self.assertEqual(samples[0].next_tool, "read_file")
            self.assertEqual(samples[1].next_tool, "exec-grep")
            self.assertEqual(samples[2].history, ["exec-pytest", "read_file"])


if __name__ == "__main__":
    unittest.main()
