"""Tests for retrieval evaluation metrics."""

import pytest

from tests.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_relevant_chunks() -> None:
    retrieved = ["chunk-3", "chunk-1", "chunk-4", "chunk-2", "chunk-5"]
    relevant = {"chunk-1", "chunk-2", "chunk-6"}

    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_recall_at_k_ignores_relevant_chunks_beyond_k() -> None:
    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    relevant = {"chunk-3", "chunk-4"}

    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        recall_at_k(["chunk-1"], {"chunk-1"}, k=k)


def test_recall_at_k_rejects_empty_relevance_set() -> None:
    with pytest.raises(ValueError, match="At least one relevant chunk"):
        recall_at_k(["chunk-1"], set())


def test_precision_at_k_measures_relevant_fraction() -> None:
    retrieved = ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
    relevant = {"chunk-1", "chunk-3"}

    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)


def test_precision_at_k_uses_only_top_k_results() -> None:
    retrieved = ["chunk-1", "chunk-2", "chunk-3"]
    relevant = {"chunk-3"}

    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)


def test_precision_at_k_handles_no_retrieved_chunks() -> None:
    assert precision_at_k([], {"chunk-1"}) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_precision_at_k_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        precision_at_k(["chunk-1"], {"chunk-1"}, k=k)


def test_precision_at_k_rejects_empty_relevance_set() -> None:
    with pytest.raises(ValueError, match="At least one relevant chunk"):
        precision_at_k(["chunk-1"], set())


def test_reciprocal_rank_rewards_earlier_relevant_results() -> None:
    retrieved = ["chunk-3", "chunk-4", "chunk-1"]
    relevant = {"chunk-1"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_none_are_relevant() -> None:
    assert reciprocal_rank(["chunk-1", "chunk-2"], {"chunk-3"}) == 0.0


def test_mean_reciprocal_rank_averages_query_scores() -> None:
    ranked_results = [
        ["chunk-1", "chunk-2"],
        ["chunk-3", "chunk-4"],
        ["chunk-5", "chunk-6"],
    ]
    relevant_chunk_sets = [
        {"chunk-1"},
        {"chunk-4"},
        {"chunk-7"},
    ]

    expected = (1.0 + 0.5 + 0.0) / 3

    assert mean_reciprocal_rank(
        ranked_results,
        relevant_chunk_sets,
    ) == pytest.approx(expected)


def test_mean_reciprocal_rank_rejects_mismatched_query_counts() -> None:
    with pytest.raises(ValueError, match="corresponding relevant chunk set"):
        mean_reciprocal_rank([["chunk-1"]], [])
