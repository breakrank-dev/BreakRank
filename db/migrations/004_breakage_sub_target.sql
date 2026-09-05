-- Migration 004: allow multiple breakages of the same kind on one symbol.
--
-- Problem: f(a, b, c) -> f(a) removes two parameters, producing two records
-- with identical (release_id, symbol_path, kind). The UNIQUE constraint from
-- migration 001 collapses them, and ON CONFLICT DO NOTHING silently drops the
-- second. Measured at roughly 7% of rows.
--
-- CLASS_REMOVED_BASE has the same collision when a class drops two bases,
-- which is why the column is named generically rather than "parameter".
--
-- sub_target discriminates within a (release, symbol, kind) group:
--   PARAMETER_* kinds   -> the parameter name
--   CLASS_REMOVED_BASE  -> the base class name
--   all other kinds     -> '' (empty string)
--
-- NOT NULL with a default, deliberately not nullable: PostgreSQL treats NULLs
-- as distinct in a unique constraint, so a nullable column would permit
-- duplicate rows and defeat the idempotency the constraint exists to provide.

ALTER TABLE breakage ADD COLUMN sub_target TEXT NOT NULL DEFAULT '';

ALTER TABLE breakage DROP CONSTRAINT breakage_release_id_symbol_path_kind_key;

ALTER TABLE breakage ADD CONSTRAINT breakage_unique_change
    UNIQUE (release_id, symbol_path, kind, sub_target);