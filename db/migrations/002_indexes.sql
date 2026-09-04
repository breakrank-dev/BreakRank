-- Indexes for the hot query:
--   package name + version -> breakages, joined to score and usage count.

-- Join breakage -> usage_index on symbol_path.
CREATE INDEX idx_breakage_symbol_path ON breakage (symbol_path);

-- Homepage: most recent releases first.
CREATE INDEX idx_release_released_at ON release (released_at DESC);

-- Most queries exclude private symbols; a partial index keeps it small.
CREATE INDEX idx_breakage_public ON breakage (release_id) WHERE is_private = FALSE;

-- Scoring job looks up predictions by model version.
CREATE INDEX idx_prediction_model_version ON prediction (model_version);