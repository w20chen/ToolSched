import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from toolsched.cli import main


class CliCommandTests(unittest.TestCase):
    def test_profile_command_does_not_require_config_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            samples_path = tmp_path / "samples.jsonl"
            out_path = tmp_path / "profiles.json"
            sample = {
                "sample_id": "s1",
                "dataset": "demo",
                "case_id": "case",
                "attempt_id": "attempt",
                "tool": "read_file",
                "operation": "file_read",
                "tool_family": "file_io",
                "timestamp": None,
                "duration_ms": 10.0,
            }
            samples_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            argv = [
                "toolsched",
                "profile",
                "--samples",
                str(samples_path),
                "--out",
                str(out_path),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["profiles"][0]["operation"], "file_read")
            self.assertEqual(payload["profiles"][0]["tool_family"], "file_io")


if __name__ == "__main__":
    unittest.main()
