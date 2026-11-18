"""Tests for the long-term memory retrieval evaluation dataset."""

from tests.evaluation.long_term_memory_retrieval_dataset import (
    LONG_TERM_MEMORY_RETRIEVAL_CASES,
)

ALLOWED_CATEGORIES = {"preference", "project", "instruction", "fact"}


def test_dataset_contains_evaluation_cases() -> None:
    assert len(LONG_TERM_MEMORY_RETRIEVAL_CASES) >= 4


def test_every_case_has_a_query_and_relevant_memories() -> None:
    for case in LONG_TERM_MEMORY_RETRIEVAL_CASES:
        assert case.query.strip()
        assert case.relevant_memory_ids


def test_relevant_memory_ids_exist_in_each_case() -> None:
    for case in LONG_TERM_MEMORY_RETRIEVAL_CASES:
        assert case.relevant_memory_ids <= case.memories.keys()


def test_memory_ids_are_unique_within_each_case() -> None:
    for case in LONG_TERM_MEMORY_RETRIEVAL_CASES:
        assert len(case.memories) == len(set(case.memories))


def test_dataset_contains_unique_queries() -> None:
    queries = [case.query for case in LONG_TERM_MEMORY_RETRIEVAL_CASES]

    assert len(queries) == len(set(queries))


def test_every_memory_has_nonempty_text() -> None:
    for case in LONG_TERM_MEMORY_RETRIEVAL_CASES:
        assert case.memories
        assert all(memory.strip() for memory in case.memories.values())


def test_dataset_covers_all_memory_categories() -> None:
    categories = {case.category for case in LONG_TERM_MEMORY_RETRIEVAL_CASES}

    assert categories == ALLOWED_CATEGORIES


def test_every_case_uses_a_supported_category() -> None:
    for case in LONG_TERM_MEMORY_RETRIEVAL_CASES:
        assert case.category in ALLOWED_CATEGORIES
