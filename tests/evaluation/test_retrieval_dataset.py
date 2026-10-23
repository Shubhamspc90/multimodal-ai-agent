"""Validate the labeled retrieval evaluation dataset."""

from tests.evaluation.retrieval_dataset import RETRIEVAL_CASES


def test_dataset_contains_evaluation_cases() -> None:
    assert len(RETRIEVAL_CASES) >= 3


def test_every_case_has_a_question_and_relevant_chunks() -> None:
    for case in RETRIEVAL_CASES:
        assert case.question.strip()
        assert case.relevant_chunk_ids


def test_relevant_chunk_ids_exist_in_the_case() -> None:
    for case in RETRIEVAL_CASES:
        assert case.relevant_chunk_ids <= case.chunks.keys()


def test_chunk_ids_are_unique_within_each_case() -> None:
    for case in RETRIEVAL_CASES:
        assert len(case.chunks) == len(set(case.chunks))


def test_dataset_contains_unique_questions() -> None:
    questions = [case.question for case in RETRIEVAL_CASES]
    assert len(questions) == len(set(questions))


def test_every_chunk_has_nonempty_text() -> None:
    for case in RETRIEVAL_CASES:
        assert case.chunks
        assert all(text.strip() for text in case.chunks.values())
