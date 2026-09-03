-- Decision 4: the pipeline stores structured facts; the API renders the
-- sentence. Drop the prose column, add a JSONB payload.
--
-- Safe to DROP here because the only rows are seed data. After week 3,
-- a change like this would need a copy-then-drop migration instead.

ALTER TABLE breakage DROP COLUMN explanation;
ALTER TABLE breakage ADD COLUMN detail JSONB NOT NULL DEFAULT '{}'::jsonb;