"""
What ships (NOTES §36.2): path length alone, shortest public import path
first, nothing fitted.

Item 15's rule, fixed and pushed before its run (§36.1), set the two
learned rankers against it at the seven reference dates of §23.5, and
neither beat it: not the model on all 17 features, and not LambdaRank on
the two reachability features. Ordering its ties by the model did not
win either. So the site serves path length's scores, and the report
quotes path length's numbers; the model's scores stay in the database
as the fallback (§28.5 says how the API picks one).

One place for what that means in code, so the loader, the stories and
the scripts that measure it cannot drift apart:

  SHIPPED_RANKER  "path", the ranker the site serves. ml/db.py --scores
                  loads it unless told --ranker model, and stories.py
                  ranks by it.
  path_version()  its model_run.version: "path-length" for label_alias,
                  the label that ships (§24), and the label's name
                  added for any other, so another label's numbers can
                  never land in the served row.
  PATH_METRICS    the file train.py writes that row to, beside
                  metrics.json, the model's.
  path_scores()   the score: minus public_depth. A change with no depth
                  scores as the deepest, of the frame it is measured on
                  (a test half, as §34.1 measured it) or, for the site,
                  of its own release (§36.1).

public_depth is the fewer of the symbol's own module depth and the
depth of its shortest export path (labels.py), so a change has no depth
only if its module depth is missing; the loader counts them.
"""

import pathlib

import pandas as pd

SHIPPED_RANKER = "path"
RANKERS = ("path", "model")
PATH_METRICS = pathlib.Path("artifacts") / "path_metrics.json"
SHIPPED_LABEL = "label_alias"

# The seven reference dates of §23.5, which `stability.py --at` sweeps
# and scripts/path_verdict.py reads (its DATES; a test keeps the two the
# same). With the sweep's own quantile dates they make the thirteen the
# report quotes.
REFERENCE_DATES = ("2025-08-07", "2025-10-06", "2025-12-03", "2026-01-18",
                   "2026-03-02", "2026-04-02", "2026-05-04")


def path_version(label: str) -> str:
    """model_run.version for path length's scores under one label."""
    return "path-length" if label == SHIPPED_LABEL else f"path-length-{label}"


def path_scores(df: pd.DataFrame, within: list[str] | None = None
                ) -> pd.Series:
    """Minus public_depth, one score per row of df, aligned to its index.

    A row with no depth scores as the deepest: of the whole frame when
    `within` is None, which is how baselines.py scores a test half
    (§34.1), or of its own group, the rows sharing the `within` columns,
    which is how the site's release lists are scored (§36.1). A group
    with no depth at all scores 0, as a frame with none does."""
    depth = pd.to_numeric(df["public_depth"], errors="coerce")
    if within is None:
        deepest = depth.max() if depth.notna().any() else 0
        filled = depth.fillna(deepest)
    else:
        keys = [df[c].astype(str) for c in within]
        deepest = depth.groupby(keys, sort=False).transform("max")
        filled = depth.fillna(deepest).fillna(0)
    # + 0.0 turns a release with no depth's -0.0 into 0.0 for the database.
    return -filled.astype(float) + 0.0
