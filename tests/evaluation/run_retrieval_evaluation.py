"""Run retrieval-quality evaluation using embeddings and PostgreSQL."""

from hashlib import sha256

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_storage import PostgresVectorStorage
from tests.evaluation.retrieval_dataset import RETRIEVAL_CASES
from tests.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def evaluate_retrieval() -> dict[str, float]:
    """Evaluate Precision@5, Recall@5, and MRR on the labeled dataset."""
    embedding_service = EmbeddingService()
    vector_storage = PostgresVectorStorage()

    ranked_results: list[list[str]] = []
    relevant_chunk_sets: list[set[str]] = []

    for case in RETRIEVAL_CASES:
        chunk_ids = list(case.chunks)
        chunk_texts = list(case.chunks.values())

        document_id = sha256(
            f"retrieval-evaluation:{case.question}".encode()
        ).hexdigest()

        embeddings = embedding_service.embed_texts(chunk_texts)

        vector_storage.store_chunks(
            document_id=document_id,
            chunks=chunk_texts,
            embeddings=embeddings,
            metadata={"evaluation": True},
        )

        query_embedding = embedding_service.embed_text(case.question)

        results = vector_storage.search_similar_chunks(
            document_id=document_id,
            query_embedding=query_embedding,
            limit=5,
        )

        ranked_chunk_ids = [chunk_ids[result["chunk_index"]] for result in results]
        relevant_chunk_ids = set(case.relevant_chunk_ids)

        ranked_results.append(ranked_chunk_ids)
        relevant_chunk_sets.append(relevant_chunk_ids)

        precision = precision_at_k(
            ranked_chunk_ids,
            relevant_chunk_ids,
            k=5,
        )
        recall = recall_at_k(
            ranked_chunk_ids,
            relevant_chunk_ids,
            k=5,
        )

        print(f"\nQuestion: {case.question}")
        print(f"Expected relevant chunks: {sorted(relevant_chunk_ids)}")
        print(f"Retrieved chunks: {ranked_chunk_ids}")
        print(f"Precision@5: {precision:.3f}")
        print(f"Recall@5: {recall:.3f}")

    precision_scores = [
        precision_at_k(ranked, relevant, k=5)
        for ranked, relevant in zip(
            ranked_results,
            relevant_chunk_sets,
            strict=True,
        )
    ]

    recall_scores = [
        recall_at_k(ranked, relevant, k=5)
        for ranked, relevant in zip(
            ranked_results,
            relevant_chunk_sets,
            strict=True,
        )
    ]

    metrics = {
        "precision_at_5": sum(precision_scores) / len(precision_scores),
        "recall_at_5": sum(recall_scores) / len(recall_scores),
        "mean_reciprocal_rank": mean_reciprocal_rank(
            ranked_results,
            relevant_chunk_sets,
        ),
    }

    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Queries evaluated: {len(RETRIEVAL_CASES)}")
    print(f"Mean Precision@5: {metrics['precision_at_5']:.3f}")
    print(f"Mean Recall@5: {metrics['recall_at_5']:.3f}")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.3f}")

    return metrics


if __name__ == "__main__":
    evaluate_retrieval()
