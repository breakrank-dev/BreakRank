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


def ndcg_at_k(df: pd.DataFrame, score: str, label: str, k: int = 20,
              gain: str | None = None) -> float:
    """Mean nDCG@k over rankable pairs (same restriction as precision@k).

    gain names a column of per-row gains to use in place of the 0/1 label:
    graded relevance (F2) scores a change 1, 3, 7 or 15 by how many
    packages use it. The pairs judged are the same either way, the ones
    rankable under the 0/1 label, so the two numbers describe one exam.
    Plain nDCG@k scores "the change 25 packages use, then the one 1 uses"
    and the reverse order the same; with gains, the first scores higher."""
    out = []
    for g in _rankable(df, label, k):
        rel = (g[gain].to_numpy(float) if gain else g[label].to_numpy(int))
        gains = _ranked(g[score].to_numpy(float), rel, k)
        disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(rel)[::-1][:k]
        idisc = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        denom = float((ideal * idisc).sum())
        out.append(float((gains * disc).sum()) / denom if denom else 0.0)
    return float(np.mean(out)) if out else 0.0


def average_precision_ties(y, s, sample_weight=None) -> float:
    """Average precision expected over every order of tied scores (F42,
    NOTES §35).

    sklearn's average_precision_score scores a run of tied scores as one
    threshold: every positive in the run is credited with the precision
    at the run's end. A score that says nothing about the order inside
    its ties should be scored as if every order of them were equally
    likely, which is how precision@10 and nDCG@20 already treat ties
    (_ranked, one fixed random order). This is that expectation, exact
    (McSherry and Najork, ECIR 2008). In a run of n rows holding r
    positives, with s rows and c positives ranked above it, the row at
    position p of the run is a positive with probability r/n, and then
    has on average c + 1 + (p - 1)(r - 1)/(n - 1) positives at or above
    it. Summed over p and divided by all positives, that is the expected
    average precision. With no ties it is the ordinary one, sklearn's.

    sample_weight holds whole numbers: a row of weight k counts as k
    copies of itself, the way intervals() uses it."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    w = (np.ones_like(y) if sample_weight is None
         else np.asarray(sample_weight, float))
    if not np.allclose(w, np.round(w)) or (w < 0).any():
        raise ValueError("sample_weight must hold whole numbers")
    keep = w > 0
    y, s, w = y[keep], s[keep], np.round(w[keep])
    positives = float((w * y).sum())
    if not positives:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    n = np.add.reduceat(w, start)
    r = np.add.reduceat(w * y, start)
    above = np.r_[0.0, np.cumsum(n)[:-1]]
    found = np.r_[0.0, np.cumsum(r)[:-1]]
    harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, int(n.sum()) + 1))]
    inv = harmonic[(above + n).astype(int)] - harmonic[above.astype(int)]
    lin = n - (above + 1) * inv
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(n > 1, (r - 1) / np.maximum(n - 1, 1), 0.0)
    each = (r / n) * ((found + 1) * inv + slope * lin)
    return float(each.sum() / positives)


def evaluate(df: pd.DataFrame, score: str, label: str = "label",
             ties: bool = False) -> dict:
    """The three numbers, ready for a `model_run` row.

    ties=True scores PR-AUC with its ties averaged
    (average_precision_ties, F42); the default is sklearn's convention,
    which every number before 10 Oct used."""
    y = df[label].to_numpy(int)
    s = df[score].to_numpy(float)
    ap = average_precision_ties if ties else average_precision_score
    return {
        "pr_auc": float(ap(y, s)) if y.sum() else 0.0,
        "precision_at_10": precision_at_k(df, score, label, 10),
        "ndcg_at_20": ndcg_at_k(df, score, label, 20),
    }


PAIR = ["package", "version_from", "version_to"]
DRAWS = 2000
BOOT_SEED = 0


def intervals(df: pd.DataFrame, label: str, model: str = "model",
              baseline: str = "popularity", draws: int = DRAWS,
              seed: int = BOOT_SEED, ties: bool = False,
              per_upgrade: bool = True) -> dict:
    """95% intervals for the numbers one scored test half reports (F11).

    Each of `draws` resamples draws the test half's upgrades (version
    pairs) with replacement, keeping every row of an upgrade together,
    and recomputes PR-AUC for the model and the baseline, their ratio
    (the lift), and precision@10 and nDCG@20 for the model. The interval
    is the middle 95% of those.

    WHOLE UPGRADES, NOT ROWS (F12). Changes in one release are used or
    ignored together, and a symbol's parameter rows share one label, so
    resampling rows would pretend there are more independent observations
    than there are, and give intervals too narrow to trust.

    THE MODEL IS NOT REFITTED. These intervals say how much a number
    would move on a different draw of upgrades to test on, the models
    held fixed. How much it moves with a different training set, or a
    different cut date, is what stability.py's range across cuts shows;
    the two are different questions and neither stands in for the other.

    Returns {name: (low, high)} for pr_auc, baseline_pr_auc, lift,
    precision_at_10 and ndcg_at_20, plus "upgrades", the number of
    version pairs resampled. A resample with no positive (or, for the two
    per-upgrade numbers, no rankable upgrade) is left out of that
    number's interval.

    ties=True scores every PR-AUC with its ties averaged (F42), a row's
    draw count standing for that many copies of it. per_upgrade=False
    leaves out precision@10 and nDCG@20 (returned as NaN), when only the
    PR-AUCs and their ratio are wanted."""
    df = df.reset_index(drop=True)
    key = df[PAIR].astype(str).agg("\x1f".join, axis=1)
    codes, uniq = pd.factorize(key)
    n_pairs = len(uniq)
    y = df[label].to_numpy(int)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_pairs, np.full(n_pairs, 1 / n_pairs),
                             size=draws)

    # Rows weighted by how often their upgrade was drawn: the same as
    # stacking that many copies of the upgrade, without building them.
    ap = {model: np.full(draws, np.nan), baseline: np.full(draws, np.nan)}
    score = average_precision_ties if ties else average_precision_score
    for d in range(draws):
        w = counts[d][codes]
        if (w * y).sum() == 0:
            continue
        for s in ap:
            ap[s][d] = score(y, df[s].to_numpy(float), sample_weight=w)

    # The per-upgrade numbers: each rankable upgrade's own value, once,
    # then averaged with the draw's counts as weights.
    per = {"precision_at_10": np.full(draws, np.nan),
           "ndcg_at_20": np.full(draws, np.nan)}
    for k, name in ((10, "precision_at_10"), (20, "ndcg_at_20")):
        if not per_upgrade:
            break
        vals, where = [], []
        for g in _rankable(df, label, k):
            if k == 10:
                vals.append(_ranked(g[model].to_numpy(float),
                                    g[label].to_numpy(int), k).sum() / k)
            else:
                vals.append(ndcg_at_k(g, model, label, k))
            where.append(codes[g.index[0]])
        c = counts[:, where].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            per[name] = (c @ np.asarray(vals, float)) / c.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        lift = ap[model] / ap[baseline]
    return {"pr_auc": _band(ap[model]),
            "baseline_pr_auc": _band(ap[baseline]), "lift": _band(lift),
            "precision_at_10": _band(per["precision_at_10"]),
            "ndcg_at_20": _band(per["ndcg_at_20"]), "upgrades": n_pairs}


def tie_averaged(df: pd.DataFrame, label: str, model: str = "model",
                 draws: int = DRAWS, seed: int = BOOT_SEED) -> dict:
    """F42 and F41 in one place (NOTES §34, §35), for one scored test half
    that carries the model's scores and the baselines (add_baseline_scores):
    the model's, popularity's and path length's PR-AUC with ties averaged,
    the model's lift over each of the two, and the paired 95% interval of
    each lift, every PR-AUC in it with ties averaged too. `path` is
    evaluate()'s dict for path length alone, so its precision@10 and
    nDCG@20 come with it. stability.py, train.py and final_eval.py all take
    their numbers from here."""
    y = df[label].to_numpy(int)
    nan = float("nan")
    mine = average_precision_ties(y, df[model]) if y.sum() else 0.0
    pop = average_precision_ties(y, df["popularity"]) if y.sum() else 0.0
    path = evaluate(df, "path", label, ties=True)
    ci_pop = intervals(df, label, model, "popularity", draws, seed,
                       ties=True, per_upgrade=False)
    ci_path = intervals(df, label, model, "path", draws, seed, ties=True,
                        per_upgrade=False)
    return {"model": mine, "popularity": pop, "path": path,
            "lift_pop": mine / pop if pop else nan,
            "lift_path": mine / path["pr_auc"] if path["pr_auc"] else nan,
            "ci_pop": ci_pop["lift"], "ci_path": ci_path["lift"]}


def _band(x: np.ndarray) -> tuple[float, float]:
    """The middle 95% of the resampled values: 2.5th to 97.5th percentile.
    Resamples with nothing to measure (NaN) are left out."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(x, [2.5, 97.5])
    return (float(lo), float(hi))


def compare(results: dict[str, dict], baseline: str | None = None) -> str:
    """A table, sorted by PR-AUC, with the lift over a named baseline."""
    t = pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
    if baseline and baseline in t.index:
        t["pr_auc_lift"] = t["pr_auc"] / t.loc[baseline, "pr_auc"]
    return t.round(4).to_string()
