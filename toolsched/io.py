from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import ToolSample


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_samples(path: Path, samples: Iterable[ToolSample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json(), ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def read_samples(path: Path) -> list[ToolSample]:
    return [ToolSample.from_json(row) for row in read_jsonl(path)]

