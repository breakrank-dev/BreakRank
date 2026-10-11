-- 007: add an 'unknown' analysis status and make it the column default.
--
-- Same reasoning as 006: a release nobody has measured is not a release with
-- zero changes. Until now, a release the loader could not classify silently
-- read 'analysed' with n_changes = 0 (232 releases on 9 Oct 2026). After this,
-- the loader writes 'unknown' for those, and the site shows "not compared".
--
-- The runner applies this whole file in one transaction, so either all three
-- statements apply or none do.

ALTER TABLE release DROP CONSTRAINT release_analysis_status_check;

ALTER TABLE release ADD CONSTRAINT release_analysis_status_check
    CHECK (analysis_status IN ('analysed', 'analysed_clean', 'analysis_failed',
                               'no_source', 'yanked', 'no_baseline', 'unknown'));

ALTER TABLE release ALTER COLUMN analysis_status SET DEFAULT 'unknown';
