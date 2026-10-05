"""Retrieval metrics on section labels. Pure functions, no API calls.

`ranked` is the list of retrieved section labels in rank order (one entry per section);
`expected` is the set of labels the question needs.
"""

import math


def mrr(ranked: list[str], expected: set[str]) -> float:
    """1 / rank of the first expected section; 0 if none is retrieved.

    Rank 1 -> 1.0, rank 2 -> 0.5, rank 5 -> 0.2. Averaged over questions, it rewards putting
    the right section first, not just somewhere in the list.
    """
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
    """Binary-relevance nDCG: rewards expected sections ranked higher.

    Each hit is worth 1 / log2(rank + 1), so rank 1 counts fully and later ranks less. Dividing
    by the best possible score gives 0 to 1, comparable across questions with 1 or 2 answers.
    """
    dcg = sum(1 / math.log2(i + 1) for i, label in enumerate(ranked[:k], start=1) if label in expected)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0

