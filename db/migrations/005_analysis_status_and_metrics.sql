-- Migration 005: four additions requested by the pipeline side.
--
-- 1. release.analysis_status
--    A release currently has no row at all unless a pair found changes in it.
--    That makes "we analysed this and it was clean" indistinguishable from
--    "we never looked" — and "safe to upgrade" is a useful answer the site
--    cannot currently give. Five states, because absence has several causes:
--      analysed        diffed, changes found
--      analysed_clean  diffed, zero changes: a real "nothing broke"
--      analysis_failed download or griffe failed; see data/failures.csv
--      no_source       wheel-only, griffe has nothing to read. Breaks
--                      introduced here are attributed to the next release
--                      that shipped source: a known via_version off-by-one.
--      yanked          deliberately skipped; nobody upgraded through it
--
-- 2. release.n_changes
--    Denormalised count so the homepage can rank releases without a GROUP BY
--    over 19,000 breakage rows on every request.
--
-- 3. breakage.inherited_by
--    How many other classes inherit a change. 0 for ~97% of rows, 3,242 for
--    the transformers ModuleUtilsMixin case. Not a model feature (measured
--    zero gain) but a strong display signal.
--
-- 4. model_run.positive_rate
--    PR-AUC's floor is the positive rate, not 0.5. Stored beside the metric
--    so the number can never be displayed without its floor. Nullable,
--    deliberately: older runs genuinely do not have one, and NOT NULL
--    DEFAULT 0 would assert a floor of zero, which is false.

ALTER TABLE release ADD COLUMN analysis_status TEXT NOT NULL DEFAULT 'analysed'
    CHECK (analysis_status IN (
        'analysed', 'analysed_clean', 'analysis_failed', 'no_source', 'yanked'
    ));

ALTER TABLE release ADD COLUMN n_changes INTEGER NOT NULL DEFAULT 0;

ALTER TABLE breakage ADD COLUMN inherited_by INTEGER NOT NULL DEFAULT 0;

ALTER TABLE model_run ADD COLUMN positive_rate REAL;

CREATE INDEX idx_release_clean
    ON release (package_id) WHERE analysis_status = 'analysed_clean';