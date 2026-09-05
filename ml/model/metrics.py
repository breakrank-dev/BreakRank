"""
How BreakRank is scored. Shared by the baselines and the ranker.

Three numbers, and the column names match Varad's `model_run` table so a
training run drops straight into the database:

    pr_auc           average precision. The book's primary metric.
    precision_at_10  of the 10 changes we put at the top of ONE upgrade,
                     how many mattered.
    ndcg_at_20       ranking quality over the top 20 of one upgrade.

NEVER accuracy. With ~4% positives, "everything is fine" scores 96% and
has learned nothing. That number is not conservative, it is wrong, and
the book bans it for exactly this reason.

precision@10 and nDCG@20 are computed PER VERSION PAIR and then averaged,
which is the only framing that matches the product. A user upgrading
pandas 2.1.0 -> 2.2.0 sees the changes in THAT upgrade ranked. Taking the
global top 10 across 23,000 rows would answer a question nobody asked and
would be dominated by whichever package happens to churn most.

Ties matter here. A baseline that gives every row the same score is not
"average" — without care it can look brilliant or terrible depending on
how the sort happens to fall. So ties are broken by a fixed random
permutation, which is what "no information" actually means.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

TIE_SEED = 0


def _ranked(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Labels of the top-k rows, ties broken at random but reproducibly."""
    rng = np.random.default_rng(TIE_SEED)
    order = np.lexsort((rng.random(len(scores)), -scores))
    return labels[order][:k]


def _rankable(df: pd.DataFrame, label: str, k: int):
    """Version pairs where ranking can actually be judged.

    A pair needs at least one positive (otherwise there is nothing to
    find) AND MORE THAN k changes (otherwise the "top k" is the whole
    release and every ordering scores the same).

    That second condition was missing at first, and it made the metric
    lie in the most flattering direction: a constant score — no ranking
    at all — measured 0.5706, identical to every real baseline, because
    most releases are smaller than 10 changes and the top-10 of a 3-row
    release is the release. The number looked like skill and was really
    just how dense positives are inside small releases.
    """
    for _, g in df.groupby(["package", "version_from", "version_to"], sort=False):
        if g[label].sum() > 0 and len(g) > k:
            yield g


def n_rankable(df: pd.DataFrame, label: str, k: int = 10) -> int:
    return sum(1 for _ in _rankable(df, label, k))


def precision_at_k(df: pd.DataFrame, score: str, label: str, k: int = 10) -> float:
    """Mean over rankable pairs of (relevant in top k) / k."""
    out = [_ranked(g[score].to_numpy(float), g[label].to_numpy(int), k).sum() / k
           for g in _rankable(df, label, k)]
    return float(np.mean(out)) if out else 0.0


def ndcg_at_k(df: pd.DataFrame, score: str, label: str, k: int = 20) -> float:
    """Mean nDCG@k over rankable pairs (same restriction as precision@k)."""
    out = []
    for g in _rankable(df, label, k):
        rel = g[label].to_numpy(int)
        gains = _ranked(g[score].to_numpy(float), rel, k)
        disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(rel)[::-1][:k]
        idisc = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        denom = float((ideal * idisc).sum())
        out.append(float((gains * disc).sum()) / denom if denom else 0.0)
    return float(np.mean(out)) if out else 0.0


def evaluate(df: pd.DataFrame, score: str, label: str = "label") -> dict:
    """The three numbers, ready for a `model_run` row."""
    y = df[label].to_numpy(int)
    s = df[score].to_numpy(float)
    return {
        "pr_auc": float(average_precision_score(y, s)) if y.sum() else 0.0,
        "precision_at_10": precision_at_k(df, score, label, 10),
        "ndcg_at_20": ndcg_at_k(df, score, label, 20),
    }


def compare(results: dict[str, dict], baseline: str | None = None) -> str:
    """A table, sorted by PR-AUC, with the lift over a named baseline."""
    t = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
    if baseline and baseline in t.index:
        t["pr_auc_lift"] = t["pr_auc"] / t.loc[baseline, "pr_auc"]
    return t.round(4).to_string()
