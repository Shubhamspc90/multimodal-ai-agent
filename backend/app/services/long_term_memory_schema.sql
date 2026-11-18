CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS long_term_memories (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL
        REFERENCES users(id)
        ON DELETE CASCADE,
    memory_text TEXT NOT NULL
        CHECK (length(trim(memory_text)) > 0),
    category TEXT NOT NULL
        CHECK (
            category IN (
                'preference',
                'project',
                'instruction',
                'fact'
            )
        ),
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_long_term_memories_user_created
    ON long_term_memories (user_id, created_at DESC);
