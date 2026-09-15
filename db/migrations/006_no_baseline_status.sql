-- Migration 006: a sixth analysis_status.
--
-- The oldest release in each package's analysis window has no preceding
-- release to diff against, so n_changes = 0 for it means "nothing to compare"
-- rather than "nothing broke". Recording that as analysed_clean would make
-- the site say "safe to upgrade" about a release it never compared.
--
-- Same principle as returning null for an unscored breakage instead of 0.0:
-- absence of a measurement is not a measurement of zero.

ALTER TABLE release DROP CONSTRAINT release_analysis_status_check;

ALTER TABLE release ADD CONSTRAINT release_analysis_status_check
    CHECK (analysis_status IN (
        'analysed', 'analysed_clean', 'analysis_failed',
        'no_source', 'yanked', 'no_baseline'
    ));