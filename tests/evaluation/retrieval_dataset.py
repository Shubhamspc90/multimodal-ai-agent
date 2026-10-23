"""Small, deterministic dataset for retrieval-quality evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    """A question and the chunks expected to support its answer."""

    question: str
    chunks: dict[str, str]
    relevant_chunk_ids: frozenset[str]


RETRIEVAL_CASES = (
    RetrievalCase(
        question="Which database stores application vectors?",
        chunks={
            "database-1": "The application stores vector embeddings in PostgreSQL.",
            "database-2": (
                "PostgreSQL uses the pgvector extension for similarity search."
            ),
            "database-3": "The web interface is built with React.",
            "database-4": "The API is implemented with FastAPI.",
            "database-5": "Unit tests are executed with pytest.",
            "database-6": "Audio files are converted into text transcripts.",
            "database-7": "Image uploads are processed by the image service.",
            "database-8": "The application uses asynchronous request handlers.",
            "database-9": "Environment variables configure application settings.",
            "database-10": "Structured logs help diagnose application failures.",
        },
        relevant_chunk_ids=frozenset({"database-1", "database-2"}),
    ),
    RetrievalCase(
        question="Which library generates text embeddings?",
        chunks={
            "embedding-1": (
                "The embedding service uses sentence-transformers/all-MiniLM-L6-v2."
            ),
            "embedding-2": "The embedding vectors contain 384 dimensions.",
            "embedding-3": "The API is implemented with FastAPI.",
            "embedding-4": "PostgreSQL stores document chunks.",
            "embedding-5": "The frontend uses React components.",
            "embedding-6": "The speech service transcribes uploaded audio.",
            "embedding-7": "Image understanding analyzes visual inputs.",
            "embedding-8": "Pytest runs automated application tests.",
            "embedding-9": "Environment variables configure service settings.",
            "embedding-10": "Application logs record runtime events.",
        },
        relevant_chunk_ids=frozenset({"embedding-1"}),
    ),
    RetrievalCase(
        question="How does the application store document chunks?",
        chunks={
            "storage-1": "Document chunks are persisted in PostgreSQL.",
            "storage-2": "The database uses pgvector for vector similarity search.",
            "storage-3": "Audio input is processed by speech-to-text.",
            "storage-4": "The image service analyzes uploaded images.",
            "storage-5": "The API uses asynchronous request handlers.",
            "storage-6": "React renders the browser-based user interface.",
            "storage-7": "Pytest executes the automated test suite.",
            "storage-8": "Environment variables configure database access.",
            "storage-9": "Structured logs capture application events.",
            "storage-10": "The speech service returns transcribed text.",
        },
        relevant_chunk_ids=frozenset({"storage-1"}),
    ),
    RetrievalCase(
        question="How is spoken audio converted into text?",
        chunks={
            "audio-1": (
                "The speech-to-text service converts uploaded audio "
                "into text transcripts."
            ),
            "audio-2": (
                "The application uses asynchronous handlers to process requests."
            ),
            "audio-3": "The image service analyzes visual inputs.",
            "audio-4": "The text embedding service converts text into vectors.",
            "audio-5": "PostgreSQL stores document chunks and embeddings.",
            "audio-6": "The API uses FastAPI for HTTP request handling.",
            "audio-7": "Pytest executes automated tests.",
            "audio-8": "The frontend uses React components.",
            "audio-9": "Environment variables configure application settings.",
            "audio-10": "Structured logs help diagnose runtime failures.",
        },
        relevant_chunk_ids=frozenset({"audio-1"}),
    ),
    RetrievalCase(
        question="Which component analyzes uploaded images?",
        chunks={
            "image-1": (
                "The image service analyzes uploaded images "
                "and interprets their visual content."
            ),
            "image-2": "The speech-to-text service transcribes audio.",
            "image-3": "The embedding service generates text vectors.",
            "image-4": "PostgreSQL persists document chunks.",
            "image-5": "FastAPI exposes the application's HTTP endpoints.",
            "image-6": "React renders the browser-based interface.",
            "image-7": "Pytest runs the automated test suite.",
            "image-8": "The application uses asynchronous request handlers.",
            "image-9": "Environment variables configure service settings.",
            "image-10": "Application logs capture runtime events.",
        },
        relevant_chunk_ids=frozenset({"image-1"}),
    ),
    RetrievalCase(
        question="Which framework handles HTTP API requests?",
        chunks={
            "api-1": "FastAPI handles HTTP requests and exposes API endpoints.",
            "api-2": "The application uses asynchronous request handlers.",
            "api-3": "React renders the user interface in the browser.",
            "api-4": "PostgreSQL stores document chunks.",
            "api-5": "The embedding service generates 384-dimensional vectors.",
            "api-6": "The speech service transcribes uploaded audio.",
            "api-7": "The image service analyzes visual inputs.",
            "api-8": "Pytest runs automated application tests.",
            "api-9": "Environment variables configure API settings.",
            "api-10": "Structured logs capture application failures.",
        },
        relevant_chunk_ids=frozenset({"api-1"}),
    ),
    RetrievalCase(
        question="Where are vector embeddings persisted?",
        chunks={
            "paraphrase-1": ("The application stores vector embeddings in PostgreSQL."),
            "paraphrase-2": (
                "PostgreSQL with pgvector supports similarity searches over embeddings."
            ),
            "paraphrase-3": (
                "The application stores ordinary relational records in PostgreSQL."
            ),
            "paraphrase-4": "The frontend renders components with React.",
            "paraphrase-5": "FastAPI handles incoming HTTP requests.",
            "paraphrase-6": (
                "The embedding service creates numerical representations of text."
            ),
            "paraphrase-7": "The application uses asynchronous request handlers.",
            "paraphrase-8": (
                "The database connection is configured using environment variables."
            ),
            "paraphrase-9": (
                "Document files are converted into extracted text before processing."
            ),
            "paraphrase-10": (
                "The application records diagnostic information in structured logs."
            ),
        },
        relevant_chunk_ids=frozenset({"paraphrase-1", "paraphrase-2"}),
    ),
    RetrievalCase(
        question="What converts spoken recordings into written text?",
        chunks={
            "speech-1": (
                "The speech-to-text service converts uploaded audio "
                "into text transcripts."
            ),
            "speech-2": (
                "The image service analyzes uploaded images and visual content."
            ),
            "speech-3": "The embedding service converts text into numerical vectors.",
            "speech-4": "The API accepts uploaded files through HTTP endpoints.",
            "speech-5": "Audio files are stored before downstream processing.",
            "speech-6": "The application uses asynchronous request handlers.",
            "speech-7": "PostgreSQL stores document chunks and vector embeddings.",
            "speech-8": "The frontend displays the results returned by the backend.",
            "speech-9": "Automated tests validate the speech service behavior.",
            "speech-10": (
                "Structured logs record errors encountered during processing."
            ),
        },
        relevant_chunk_ids=frozenset({"speech-1"}),
    ),
    RetrievalCase(
        question="Which component provides the HTTP API?",
        chunks={
            "framework-1": ("FastAPI handles HTTP requests and exposes API endpoints."),
            "framework-2": "The application uses asynchronous request handlers.",
            "framework-3": "The frontend uses React to render browser components.",
            "framework-4": "The database stores document chunks and embeddings.",
            "framework-5": "The embedding service generates 384-dimensional vectors.",
            "framework-6": "The speech service transcribes uploaded audio.",
            "framework-7": "The image service analyzes visual inputs.",
            "framework-8": "Pytest runs the automated application test suite.",
            "framework-9": "Environment variables configure service settings.",
            "framework-10": "Structured logs capture application failures.",
        },
        relevant_chunk_ids=frozenset({"framework-1"}),
    ),
    RetrievalCase(
        question="How does the system represent text for semantic search?",
        chunks={
            "paraphrase-hard-1": (
                "The embedding service converts text into numerical vectors "
                "using sentence-transformers/all-MiniLM-L6-v2."
            ),
            "paraphrase-hard-2": (
                "The embedding model produces normalized vectors with 384 dimensions."
            ),
            "paraphrase-hard-3": (
                "PostgreSQL stores document chunks and their vector embeddings."
            ),
            "paraphrase-hard-4": (
                "The image service interprets visual content from uploaded images."
            ),
            "paraphrase-hard-5": (
                "The speech-to-text service converts audio into transcripts."
            ),
            "paraphrase-hard-6": (
                "FastAPI exposes endpoints for application requests."
            ),
            "paraphrase-hard-7": (
                "The frontend renders interactive components with React."
            ),
            "paraphrase-hard-8": (
                "Pytest executes automated tests for the application."
            ),
            "paraphrase-hard-9": (
                "Environment variables configure application services."
            ),
            "paraphrase-hard-10": (
                "Structured logs record events and application failures."
            ),
        },
        relevant_chunk_ids=frozenset({"paraphrase-hard-1", "paraphrase-hard-2"}),
    ),
    RetrievalCase(
        question=(
            "How are document passages converted into searchable vectors and persisted?"
        ),
        chunks={
            "multistep-1": (
                "The embedding service converts text chunks into numerical "
                "vector representations."
            ),
            "multistep-2": (
                "PostgreSQL with pgvector stores embeddings and supports "
                "similarity search."
            ),
            "multistep-3": "The speech service converts spoken audio into text.",
            "multistep-4": "The image service analyzes uploaded visual inputs.",
            "multistep-5": "FastAPI handles HTTP requests and exposes endpoints.",
            "multistep-6": "React renders the browser-based user interface.",
            "multistep-7": "Pytest runs the automated test suite.",
            "multistep-8": "Environment variables configure database connections.",
            "multistep-9": "Structured logs capture application events.",
            "multistep-10": (
                "Document text is split into overlapping chunks before embedding."
            ),
        },
        relevant_chunk_ids=frozenset({"multistep-1", "multistep-2", "multistep-10"}),
    ),
    RetrievalCase(
        question="Which component transcribes uploaded audio recordings?",
        chunks={
            "hard-negative-1": (
                "The speech-to-text service converts uploaded audio into "
                "text transcripts."
            ),
            "hard-negative-2": ("Audio files are stored before downstream processing."),
            "hard-negative-3": ("The image service analyzes images uploaded by users."),
            "hard-negative-4": (
                "The embedding service converts text into numerical vectors."
            ),
            "hard-negative-5": (
                "The application accepts uploaded files through HTTP endpoints."
            ),
            "hard-negative-6": (
                "PostgreSQL persists document chunks and vector embeddings."
            ),
            "hard-negative-7": (
                "The application uses asynchronous handlers to process requests."
            ),
            "hard-negative-8": (
                "The frontend displays results returned by the backend."
            ),
            "hard-negative-9": ("Automated tests validate speech-processing behavior."),
            "hard-negative-10": (
                "Structured logs record errors during file processing."
            ),
        },
        relevant_chunk_ids=frozenset({"hard-negative-1"}),
    ),
    RetrievalCase(
        question="Where does semantic similarity search over document vectors run?",
        chunks={
            "similar-tech-1": (
                "PostgreSQL uses the pgvector extension for vector similarity search."
            ),
            "similar-tech-2": (
                "The embedding service generates 384-dimensional text vectors."
            ),
            "similar-tech-3": ("The application stores document chunks in PostgreSQL."),
            "similar-tech-4": (
                "FastAPI routes incoming requests to application services."
            ),
            "similar-tech-5": "The speech service transcribes audio recordings.",
            "similar-tech-6": "The image service interprets uploaded images.",
            "similar-tech-7": "React renders components in the browser.",
            "similar-tech-8": "Pytest runs automated application tests.",
            "similar-tech-9": "Environment variables configure database access.",
            "similar-tech-10": "Structured logs help diagnose runtime errors.",
        },
        relevant_chunk_ids=frozenset({"similar-tech-1"}),
    ),
)
