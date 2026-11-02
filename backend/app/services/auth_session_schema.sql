CREATE TABLE IF NOT EXISTS auth_sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL
        REFERENCES users(id)
        ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE
        CHECK (length(token_hash) = 64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    CONSTRAINT ck_auth_sessions_expiry
        CHECK (expires_at > created_at)
);

CREATE INDEX IF NOT EXISTS ix_auth_sessions_user_expires
    ON auth_sessions (user_id, expires_at);