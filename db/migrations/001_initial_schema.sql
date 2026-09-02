-- BreakRank initial schema
-- Order matters: parents before children (foreign keys).

CREATE TABLE package (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    download_rank INTEGER,
    github_repo   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE release (
    id          SERIAL PRIMARY KEY,
    package_id  INTEGER NOT NULL REFERENCES package(id) ON DELETE CASCADE,
    version     TEXT NOT NULL,
    released_at TIMESTAMPTZ,
    bump_type   TEXT CHECK (bump_type IN ('major', 'minor', 'patch', 'other')),
    UNIQUE (package_id, version)
);

CREATE TABLE breakage (
    id            SERIAL PRIMARY KEY,
    release_id    INTEGER NOT NULL REFERENCES release(id) ON DELETE CASCADE,
    symbol_path   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    is_private    BOOLEAN NOT NULL DEFAULT FALSE,
    module_depth  INTEGER,
    is_top_level  BOOLEAN,
    in_dunder_all BOOLEAN,
    explanation   TEXT,
    UNIQUE (release_id, symbol_path, kind)
);

CREATE TABLE usage_index (
    symbol_path TEXT PRIMARY KEY,
    user_count  INTEGER NOT NULL DEFAULT 0,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_run (
    id              SERIAL PRIMARY KEY,
    version         TEXT NOT NULL UNIQUE,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    pr_auc          REAL,
    precision_at_10 REAL,
    ndcg_at_20      REAL,
    notes           TEXT
);

CREATE TABLE prediction (
    breakage_id   INTEGER NOT NULL REFERENCES breakage(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL REFERENCES model_run(version) ON DELETE CASCADE,
    score         REAL NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (breakage_id, model_version)
);