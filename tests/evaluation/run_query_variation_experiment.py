"""Compare original and reformulated queries across the retrieval dataset."""

from hashlib import sha256

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_storage import PostgresVectorStorage
from tests.evaluation.retrieval_dataset import RETRIEVAL_CASES

REFORMULATIONS = {
    "Which database stores application vectors?": (
        "Where are application vector embeddings stored?"
    ),
    "Which library generates text embeddings?": (
        "What software model converts text into embedding vectors?"
    ),
    "How does the application store document chunks?": (
        "Where and how are document chunks persisted?"
    ),
    "How is spoken audio converted into text?": (
        "Which process transcribes speech into written text?"
    ),
    "Which component analyzes uploaded images?": (
        "What service interprets the contents of an uploaded image?"
    ),
    "Which framework handles HTTP API requests?": (
        "What web framework receives and processes API requests?"
    ),
    "Where are vector embeddings persisted?": (
        "Which database stores the generated embedding vectors?"
    ),
    "What converts spoken recordings into written text?": (
        "Which service produces transcripts from audio recordings?"
    ),
    "Which component provides the HTTP API?": (
        "What backend framework exposes the application's HTTP endpoints?"
    ),
    "How does the system represent text for semantic search?": (
        "How is text converted into numerical vectors for semantic search?"
    ),
    "How are document passages converted into searchable vectors and persisted?": (
        "How does the application generate embeddings from document "
        "chunks and store them for vector search?"
    ),
    "Which component transcribes uploaded audio recordings?": (
        "Which service converts uploaded audio into text transcripts?"
    ),
    "Where does semantic similarity search over document vectors run?": (
        "Which database extension performs vector similarity searches?"
    ),
}


def reciprocal_rank(ranked_ids: list[str], relevant_ids: frozenset[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_metrics(
    ranked_ids: list[str],
    relevant_ids: frozenset[str],
) -> dict[str, float]:
    """Calculate retrieval metrics for one query."""
    metrics = {}

    for cutoff in (3, 5):
        retrieved = ranked_ids[:cutoff]
        relevant_count = sum(chunk_id in relevant_ids for chunk_id in retrieved)
        metrics[f"precision@{cutoff}"] = relevant_count / cutoff
        metrics[f"recall@{cutoff}"] = relevant_count / len(relevant_ids)

    metrics["mrr"] = reciprocal_rank(ranked_ids, relevant_ids)
    return metrics


def main() -> None:
    """Compare baseline and reformulated queries over all evaluation cases."""
    if len(REFORMULATIONS) != len(RETRIEVAL_CASES):
        raise ValueError(
            "The number of reformulations must match the retrieval dataset."
        )

    dataset_questions = {case.question for case in RETRIEVAL_CASES}
    reformulation_questions = set(REFORMULATIONS)

    if dataset_questions != reformulation_questions:
        missing = sorted(dataset_questions - reformulation_questions)
        unexpected = sorted(reformulation_questions - dataset_questions)
        raise ValueError(
            f"Reformulation keys do not match the dataset. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    embedding_service = EmbeddingService()
    vector_storage = PostgresVectorStorage()

    metric_names = (
        "precision@3",
        "recall@3",
        "precision@5",
        "recall@5",
        "mrr",
    )
    totals = {
        label: dict.fromkeys(metric_names, 0.0)
        for label in ("baseline", "reformulated")
    }

    for case in RETRIEVAL_CASES:
        document_id = sha256(
            f"retrieval-evaluation:{case.question}".encode()
        ).hexdigest()
        chunk_ids = list(case.chunks)

        for label, query in (
            ("baseline", case.question),
            ("reformulated", REFORMULATIONS[case.question]),
        ):
            embedding = embedding_service.embed_text(query)
            results = vector_storage.search_similar_chunks(
                document_id=document_id,
                query_embedding=embedding,
                limit=len(chunk_ids),
            )
            ranked_ids = [chunk_ids[result["chunk_index"]] for result in results]
            metrics = calculate_metrics(ranked_ids, case.relevant_chunk_ids)

            for metric, value in metrics.items():
                totals[label][metric] += value

            relevant_ranks = [
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in case.relevant_chunk_ids
            ]
            print(
                f"{case.question}\n  {label}: relevant ranks={relevant_ranks or 'none'}"
            )

    count = len(RETRIEVAL_CASES)
    print(f"\nMEAN METRICS ACROSS {count} QUERIES")
    print(f"{'Metric':<15}{'Baseline':>12}{'Reformulated':>16}{'Change':>12}")

    for metric in metric_names:
        baseline = totals["baseline"][metric] / count
        reformulated = totals["reformulated"][metric] / count
        print(
            f"{metric:<15}{baseline:>12.3f}"
            f"{reformulated:>16.3f}{reformulated - baseline:>+12.3f}"
        )


if __name__ == "__main__":
    main()
