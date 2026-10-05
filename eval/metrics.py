"""Retrieval metrics on section labels. Pure functions, no API calls.

`ranked` is the list of retrieved section labels in rank order (one entry per section);
`expected` is the set of labels the question needs.
"""

import math


def mrr(ranked: list[str], expected: set[str]) -> float:
    """1 / rank of the first expected section; 0 if none is retrieved."""
    for i, label in enumerate(ranked, start=1):
        if label in expected:
            return 1 / i
    return 0.0


def recall_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Share of expected sections found in the top k."""
    return len(expected & set(ranked[:k])) / len(expected) if expected else 0.0


def precision_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Share of the top k that are expected. Max is len(expected)/k (0.20 at k=5 for one section)."""
    return len(expected & set(ranked[:k])) / k


def ndcg_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Binary-relevance nDCG: rewards expected sections ranked higher."""
    dcg = sum(1 / math.log2(i + 1) for i, label in enumerate(ranked[:k], start=1) if label in expected)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0

