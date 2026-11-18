"""Evaluation dataset for semantic long-term memory retrieval."""

from dataclasses import dataclass
from typing import Literal

MemoryCategory = Literal["preference", "project", "instruction", "fact"]


@dataclass(frozen=True)
class LongTermMemoryRetrievalCase:
    """A query and candidate memories with labeled relevant memory IDs."""

    query: str
    category: MemoryCategory
    memories: dict[str, str]
    relevant_memory_ids: frozenset[str]


LONG_TERM_MEMORY_RETRIEVAL_CASES = (
    LongTermMemoryRetrievalCase(
        query="Which programming language does the user prefer?",
        category="preference",
        memories={
            "preference-1": "Prefers writing backend applications in Python.",
            "preference-2": "Prefers concise technical explanations.",
            "preference-3": "Enjoys working with relational databases.",
            "preference-4": "Uses JavaScript for frontend development.",
            "preference-5": "Prefers automated testing for software projects.",
        },
        relevant_memory_ids=frozenset({"preference-1"}),
    ),
    LongTermMemoryRetrievalCase(
        query="What kind of technical explanations are preferred?",
        category="preference",
        memories={
            "preference-1": "Prefers short and clear technical explanations.",
            "preference-2": (
                "Prefers detailed examples when learning technical concepts."
            ),
            "preference-3": "Uses PostgreSQL for relational data storage.",
            "preference-4": "Works on an artificial intelligence application.",
            "preference-5": "Prefers incremental changes to existing code.",
        },
        relevant_memory_ids=frozenset({"preference-1", "preference-2"}),
    ),
    LongTermMemoryRetrievalCase(
        query=("Which application is currently being developed?"),
        category="project",
        memories={
            "project-1": "Developing an AI-powered research assistant.",
            "project-2": "Building a web application for managing events.",
            "project-3": "Creating a document search and retrieval system.",
            "project-4": "Maintaining an application for employee attendance.",
            "project-5": "Developing a desktop inventory management system.",
        },
        relevant_memory_ids=frozenset({"project-1"}),
    ),
    LongTermMemoryRetrievalCase(
        query=("What technology is used for vector storage in the AI project?"),
        category="project",
        memories={
            "project-1": "The AI application uses PostgreSQL with pgvector.",
            "project-2": "The frontend application is built with React.",
            "project-3": "The backend exposes APIs using FastAPI.",
            "project-4": "The project uses PostgreSQL for vector embeddings.",
            "project-5": "The application uses Redis for caching.",
        },
        relevant_memory_ids=frozenset({"project-1", "project-4"}),
    ),
    LongTermMemoryRetrievalCase(
        query=(
            "How should existing software functionality be handled during development?"
        ),
        category="instruction",
        memories={
            "instruction-1": (
                "Preserve existing working features when implementing "
                "new functionality."
            ),
            "instruction-2": (
                "Make changes incrementally and validate them with appropriate tests."
            ),
            "instruction-3": "Use meaningful Git commits for completed changes.",
            "instruction-4": "Avoid unnecessary dependency updates.",
            "instruction-5": "Document important architectural decisions.",
        },
        relevant_memory_ids=frozenset({"instruction-1", "instruction-2"}),
    ),
    LongTermMemoryRetrievalCase(
        query="What validation is expected after changing application code?",
        category="instruction",
        memories={
            "instruction-1": "Run relevant automated tests after implementation.",
            "instruction-2": "Check code formatting and linting before completion.",
            "instruction-3": "Avoid unrelated changes outside the requested scope.",
            "instruction-4": "Use feature branches for isolated development work.",
            "instruction-5": "Preserve existing database records during development.",
        },
        relevant_memory_ids=frozenset({"instruction-1", "instruction-2"}),
    ),
    LongTermMemoryRetrievalCase(
        query="Which database extension provides vector similarity search?",
        category="fact",
        memories={
            "fact-1": "The PostgreSQL database uses the pgvector extension.",
            "fact-2": "The application backend is implemented with Python.",
            "fact-3": "The frontend uses a component-based architecture.",
            "fact-4": "Automated tests are written using pytest.",
            "fact-5": "The application supports document uploads.",
        },
        relevant_memory_ids=frozenset({"fact-1"}),
    ),
    LongTermMemoryRetrievalCase(
        query=("How many dimensions are generated by the configured embedding model?"),
        category="fact",
        memories={
            "fact-1": (
                "The configured text embedding model generates 384-dimensional vectors."
            ),
            "fact-2": "The database stores application data in PostgreSQL.",
            "fact-3": "The API uses asynchronous request handling.",
            "fact-4": "The application supports audio transcription.",
            "fact-5": "The frontend uses TypeScript.",
        },
        relevant_memory_ids=frozenset({"fact-1"}),
    ),
)
