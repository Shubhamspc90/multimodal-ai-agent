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

EVALUATION_CUTOFFS = (3, 5)


def evaluate_retrieval() -> dict[str, float]:
    """Compare Precision@K and Recall@K at K=3 and K=5, plus MRR."""
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

        # Retrieve once at the largest cutoff for a fair comparison.
        results = vector_storage.search_similar_chunks(
            document_id=document_id,
            query_embedding=query_embedding,
            limit=max(EVALUATION_CUTOFFS),
        )

        ranked_chunk_ids = [chunk_ids[result["chunk_index"]] for result in results]
        relevant_chunk_ids = set(case.relevant_chunk_ids)

        ranked_results.append(ranked_chunk_ids)
        relevant_chunk_sets.append(relevant_chunk_ids)

        print(f"\nQuestion: {case.question}")
        print(f"Expected relevant chunks: {sorted(relevant_chunk_ids)}")
        print("Retrieved ranked chunks:")
        for rank, result in enumerate(results, start=1):
            chunk_id = chunk_ids[result["chunk_index"]]
            relevance = "RELEVANT" if chunk_id in relevant_chunk_ids else "NOT RELEVANT"
            print(
                f"  {rank}. {chunk_id} | "
                f"distance={result['distance']:.4f} | {relevance}"
            )

        for k in EVALUATION_CUTOFFS:
            precision = precision_at_k(
                ranked_chunk_ids,
                relevant_chunk_ids,
                k=k,
            )
            recall = recall_at_k(
                ranked_chunk_ids,
                relevant_chunk_ids,
                k=k,
            )

            print(f"Precision@{k}: {precision:.3f}")
            print(f"Recall@{k}: {recall:.3f}")

    metrics: dict[str, float] = {}

    for k in EVALUATION_CUTOFFS:
        precision_scores = [
            precision_at_k(ranked, relevant, k=k)
            for ranked, relevant in zip(
                ranked_results,
                relevant_chunk_sets,
                strict=True,
            )
        ]

        recall_scores = [
            recall_at_k(ranked, relevant, k=k)
            for ranked, relevant in zip(
                ranked_results,
                relevant_chunk_sets,
                strict=True,
            )
        ]

        metrics[f"precision_at_{k}"] = sum(precision_scores) / len(precision_scores)
        metrics[f"recall_at_{k}"] = sum(recall_scores) / len(recall_scores)

    metrics["mean_reciprocal_rank"] = mean_reciprocal_rank(
        ranked_results,
        relevant_chunk_sets,
    )

    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Queries evaluated: {len(RETRIEVAL_CASES)}")

    for k in EVALUATION_CUTOFFS:
        print(f"Mean Precision@{k}: {metrics[f'precision_at_{k}']:.3f}")
        print(f"Mean Recall@{k}: {metrics[f'recall_at_{k}']:.3f}")

    print(f"MRR: {metrics['mean_reciprocal_rank']:.3f}")

    return metrics


if __name__ == "__main__":
    evaluate_retrieval()
