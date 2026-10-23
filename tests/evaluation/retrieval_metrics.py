"""Metrics for evaluating document retrieval quality."""

from collections.abc import Sequence


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
    k: int = 5,
) -> float:
    """Return the fraction of relevant chunks found in the top k results."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_chunk_ids:
        raise ValueError("At least one relevant chunk is required.")

    retrieved_top_k = retrieved_chunk_ids[:k]
    found_relevant = set(retrieved_top_k) & relevant_chunk_ids

    return len(found_relevant) / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    if not relevant_chunk_ids:
        raise ValueError("At least one relevant chunk is required.")

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_results: Sequence[Sequence[str]],
    relevant_chunk_sets: Sequence[set[str]],
) -> float:
    """Return the mean reciprocal rank across multiple queries."""
    if not ranked_results:
        raise ValueError("At least one query is required.")

    if len(ranked_results) != len(relevant_chunk_sets):
        raise ValueError("Each query must have a corresponding relevant chunk set.")

    scores = [
        reciprocal_rank(results, relevant)
        for results, relevant in zip(ranked_results, relevant_chunk_sets, strict=True)
    ]

    return sum(scores) / len(scores)


def precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
    k: int = 5,
) -> float:
    """Return the fraction of the top k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_chunk_ids:
        raise ValueError("At least one relevant chunk must be provided.")

    retrieved_top_k = retrieved_chunk_ids[:k]

    if not retrieved_top_k:
        return 0.0

    found_relevant = set(retrieved_top_k) & relevant_chunk_ids

    return len(found_relevant) / len(retrieved_top_k)
