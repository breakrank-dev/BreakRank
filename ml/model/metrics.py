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
how the sort happens to fall. So every number is averaged over every
order of the tied scores, exactly: PR-AUC by average_precision_ties
(F42) and precision@10 and nDCG@20 by _expected (F44). That is the
convention CONVENTION names, stamped on every sweep row and model_run
note from 11 Oct (NOTES §36.2). Before it, PR-AUC took scikit-learn's
convention and the two within-upgrade numbers one fixed random order of
the ties (_ranked). Both stay, as ties=False and expected=False, to
measure an earlier number again the way it was measured.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

TIE_SEED = 0

# The convention every number takes from 11 Oct (NOTES §36.2): ties
# averaged over every order, PR-AUC (F42) and the within-upgrade numbers
# (F44) alike. Stamped on sweep rows and in model_run's notes, so no
# number is compared with one measured the other way.
CONVENTION = "ties-averaged"


def _ranked(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Labels of the top-k rows, ties broken at random but reproducibly."""
    rng = np.random.default_rng(TIE_SEED)
    order = np.lexsort((rng.random(len(scores)), -scores))
    return labels[order][:k]


def _expected(scores: np.ndarray, gains: np.ndarray, k: int) -> np.ndarray:
    """What each of the top k places holds on average over every order of
    the tied scores (F44, NOTES §36): a run of tied rows fills its places
    in every order equally often, so each of those places holds the run's
    mean gain. Exact, and the same whatever order the rows came in.

    _ranked breaks ties by one fixed random order instead, keyed by a
    row's position in its upgrade, so on a ranking that ties a lot, path
    length's, precision@10 and nDCG@20 are one draw of many."""
    scores = np.asarray(scores, dtype=float)
    gains = np.asarray(gains, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, g = scores[order], gains[order]
    run = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    mean = np.bincount(run, weights=g) / np.bincount(run)
    return mean[run][:k]


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


def precision_at_k(df: pd.DataFrame, score: str, label: str, k: int = 10,
                   expected: bool = True) -> float:
    """Mean over rankable pairs of (relevant in top k) / k, averaged over
    every order of the ties (_expected, F44). expected=False takes
    _ranked's one fixed order instead, as every number before 11 Oct
    did."""
    top = _expected if expected else _ranked
    out = [top(g[score].to_numpy(float), g[label].to_numpy(int), k).sum() / k
           for g in _rankable(df, label, k)]
    return float(np.mean(out)) if out else 0.0


def ndcg_at_k(df: pd.DataFrame, score: str, label: str, k: int = 20,
              gain: str | None = None, expected: bool = True) -> float:
    """Mean nDCG@k over rankable pairs (same restriction as precision@k).

    gain names a column of per-row gains to use in place of the 0/1 label:
    graded relevance (F2) scores a change 1, 3, 7 or 15 by how many
    packages use it. The pairs judged are the same either way, the ones
    rankable under the 0/1 label, so the two numbers describe one exam.
    Plain nDCG@k scores "the change 25 packages use, then the one 1 uses"
    and the reverse order the same; with gains, the first scores higher.

    It is averaged over every order of the ties (_expected, F44): the
    ideal ranking is fixed, so the mean of nDCG is the mean DCG over the
    ideal. expected=False takes _ranked's one fixed order instead, as
    every number before 11 Oct did."""
    out = []
    top = _expected if expected else _ranked
    for g in _rankable(df, label, k):
        rel = (g[gain].to_numpy(float) if gain else g[label].to_numpy(int))
        gains = top(g[score].to_numpy(float), rel, k)
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
    likely, which is how precision@10 and nDCG@20 treat ties too
    (_expected, F44). This is that expectation, exact
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
             ties: bool = True, expected: bool = True) -> dict:
    """The three numbers, ready for a `model_run` row, by CONVENTION:
    PR-AUC with its ties averaged (average_precision_ties, F42), and
    precision@10 and nDCG@20 as their average over every order of the
    ties (F44).

    ties=False takes scikit-learn's convention for PR-AUC, and
    expected=False _ranked's one fixed order for the other two, the way
    every number before 11 Oct was measured (NOTES §36.2)."""
    y = df[label].to_numpy(int)
    s = df[score].to_numpy(float)
    ap = average_precision_ties if ties else average_precision_score
    return {
        "pr_auc": float(ap(y, s)) if y.sum() else 0.0,
        "precision_at_10": precision_at_k(df, score, label, 10,
                                          expected=expected),
        "ndcg_at_20": ndcg_at_k(df, score, label, 20, expected=expected),
    }


PAIR = ["package", "version_from", "version_to"]
DRAWS = 2000
BOOT_SEED = 0


def _resampled_ap(df: pd.DataFrame, label: str, names: list[str],
                  draws: int, seed: int, ties: bool):
    """PR-AUC of each named score on `draws` resamples of the test half's
    upgrades, drawn with replacement, every row of an upgrade kept
    together. Returns ({name: one value per draw}, the draw counts, each
    row's upgrade code, the number of upgrades). The draws depend on the
    seed and the upgrades alone, so every score named here, and every
    call with the same seed on the same rows, is scored on the same
    resamples: that is what makes a ratio of two of them paired."""
    key = df[PAIR].astype(str).agg("\x1f".join, axis=1)
    codes, uniq = pd.factorize(key)
    n_pairs = len(uniq)
    y = df[label].to_numpy(int)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_pairs, np.full(n_pairs, 1 / n_pairs),
                             size=draws)

    # Rows weighted by how often their upgrade was drawn: the same as
    # stacking that many copies of the upgrade, without building them.
    ap = {s: np.full(draws, np.nan) for s in dict.fromkeys(names)}
    score = average_precision_ties if ties else average_precision_score
    cols = {s: df[s].to_numpy(float) for s in ap}
    for d in range(draws):
        w = counts[d][codes]
        if (w * y).sum() == 0:
            continue
        for s in ap:
            ap[s][d] = score(y, cols[s], sample_weight=w)
    return ap, counts, codes, n_pairs


def intervals(df: pd.DataFrame, label: str, model: str = "model",
              baseline: str = "popularity", draws: int = DRAWS,
              seed: int = BOOT_SEED, ties: bool = True,
              per_upgrade: bool = True, expected: bool = True) -> dict:
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

    Every PR-AUC takes its ties averaged (F42), a row's draw count
    standing for that many copies of it, and precision@10 and nDCG@20
    their average over every order of the ties (F44): CONVENTION.
    ties=False and expected=False measure the old way. per_upgrade=False
    leaves out precision@10 and nDCG@20 (returned as NaN), when only the
    PR-AUCs and their ratio are wanted."""
    df = df.reset_index(drop=True)
    ap, counts, codes, n_pairs = _resampled_ap(df, label, [model, baseline],
                                               draws, seed, ties)

    # The per-upgrade numbers: each rankable upgrade's own value, once,
    # then averaged with the draw's counts as weights.
    per = {"precision_at_10": np.full(draws, np.nan),
           "ndcg_at_20": np.full(draws, np.nan)}
    top = _expected if expected else _ranked
    for k, name in ((10, "precision_at_10"), (20, "ndcg_at_20")):
        if not per_upgrade:
            break
        vals, where = [], []
        for g in _rankable(df, label, k):
            if k == 10:
                vals.append(top(g[model].to_numpy(float),
                                g[label].to_numpy(int), k).sum() / k)
            else:
                vals.append(ndcg_at_k(g, model, label, k, expected=expected))
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
    """F42 and F41 in one place (NOTES §34, §35, §36.2), for one scored
    test half that carries the model's scores and the baselines
    (add_baseline_scores): the model's, popularity's and path length's
    PR-AUC with ties averaged; the model's lift over each of the two, and
    path length's over popularity, the lift the report quotes now that
    path length ships; and the paired 95% interval of each lift, every
    PR-AUC in it with ties averaged too. All three are scored on the same
    resamples, the ones intervals() draws with the same seed. `path` is
    evaluate()'s dict for path length alone, so its precision@10 and
    nDCG@20 come with it. stability.py, train.py and final_eval.py all
    take their numbers from here."""
    y = df[label].to_numpy(int)
    nan = float("nan")
    mine = average_precision_ties(y, df[model]) if y.sum() else 0.0
    pop = average_precision_ties(y, df["popularity"]) if y.sum() else 0.0
    path = evaluate(df, "path", label, ties=True, expected=True)
    ap, _, _, _ = _resampled_ap(df.reset_index(drop=True), label,
                                [model, "popularity", "path"], draws, seed,
                                ties=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci_pop = _band(ap[model] / ap["popularity"])
        ci_path = _band(ap[model] / ap["path"])
        ci_path_pop = _band(ap["path"] / ap["popularity"])
    return {"model": mine, "popularity": pop, "path": path,
            "lift_pop": mine / pop if pop else nan,
            "lift_path": mine / path["pr_auc"] if path["pr_auc"] else nan,
            "path_pop": path["pr_auc"] / pop if pop else nan,
            "ci_pop": ci_pop, "ci_path": ci_path,
            "ci_path_pop": ci_path_pop}


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
