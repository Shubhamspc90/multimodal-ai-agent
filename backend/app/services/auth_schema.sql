CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL
        CHECK (length(trim(name)) BETWEEN 1 AND 100),
    email TEXT NOT NULL
        CHECK (length(trim(email)) BETWEEN 3 AND 320),
    password_hash TEXT NOT NULL
        CHECK (length(trim(password_hash)) BETWEEN 1 AND 500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_lower
    ON users (LOWER(email));