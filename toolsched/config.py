"""Config file loader for ToolSched.

Reads a JSON config file (e.g. configs/default.json) and returns the parsed
dictionary. Intended for use by the CLI to set defaults for --datasets and
--include-dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file and return its contents as a dict."""
    import json

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
