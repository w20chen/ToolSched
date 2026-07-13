from __future__ import annotations

import random
from collections import defaultdict

from ..schema import ToolSample


def split_by_case(samples: list[ToolSample], test_ratio: float = 0.2, seed: int = 7) -> tuple[list[ToolSample], list[ToolSample]]:
    cases = sorted({(s.dataset, s.case_id) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(cases)
    test_n = max(1, int(len(cases) * test_ratio)) if cases else 0
    test_cases = set(cases[:test_n])
    train, test = [], []
    for s in samples:
        if (s.dataset, s.case_id) in test_cases:
            test.append(s)
        else:
            train.append(s)
    return train, test


def split_by_dataset(samples: list[ToolSample], holdout_dataset: str) -> tuple[list[ToolSample], list[ToolSample]]:
    train = [s for s in samples if s.dataset != holdout_dataset]
    test = [s for s in samples if s.dataset == holdout_dataset]
    return train, test

