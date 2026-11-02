CREATE TABLE IF NOT EXISTS auth_sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL
        REFERENCES users(id)
        ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE
        CHECK (length(token_hash) = 64),
    csrf_token_hash TEXT
        CHECK (
            csrf_token_hash IS NULL
            OR csrf_token_hash ~ '^[0-9a-f]{64}$'
        ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    CONSTRAINT ck_auth_sessions_expiry
        CHECK (expires_at > created_at)
);

CREATE INDEX IF NOT EXISTS ix_auth_sessions_user_expires
    ON auth_sessions (user_id, expires_at);

-- Safely add CSRF token storage to existing installations.
ALTER TABLE auth_sessions
    ADD COLUMN IF NOT EXISTS csrf_token_hash TEXT
    CHECK (
        csrf_token_hash IS NULL
        OR csrf_token_hash ~ '^[0-9a-f]{64}$'
    );

-- Existing sessions have no CSRF token.
-- Revoke them so users must sign in again and receive a new session.
UPDATE auth_sessions
SET revoked_at = CURRENT_TIMESTAMP
WHERE csrf_token_hash IS NULL
  AND revoked_at IS NULL;
