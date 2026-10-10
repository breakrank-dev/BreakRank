"""
Item 4 of the fix list (F9, F10): the report's two findings, measured
again on today's data, with intervals, and worded by rules fixed before
the run (NOTES §26.1).

    python scripts/item4_findings.py

Reads  data/features.csv          the dev rows: version strings already
                                  dropped (F1), the holdout never in it
       artifacts/importance.csv   the shipped model's gain table, if there
       artifacts/metrics.json     which model that table belongs to
Writes data/item4_findings.csv. Nothing else is touched.

THE TWO FINDINGS, AS THE REPORT HAS THEM (NOTES §20.2)

  1. was_deprecated_before. A symbol deprecated before it was removed is
     used downstream 3.60% of the time, against 3.69% for the rest, over
     278 rows: "zero signal".
  2. prior_breaks_in_module is U-shaped: 2.58% with no earlier breaks in
     the module, 8.21% and 8.69% with 1-5 and 6-20, 0.25% with 21 or
     more. "Middle buckets are 3.2-3.4x the quiet bucket."

WHAT THE AUDIT FOUND WRONG WITH THEM

  F9   278 rows with 10 positives cannot tell "no effect" from an effect
       of two points either way. "Zero signal" claims more than that.
  F10  The middle peak was mostly version strings (F1). Without them the
       middle buckets were 3.92% and 2.76% against 2.30%. What held was
       the collapse at 21+.

HOW THIS COUNTS

  Once per CHANGE, not per row. A change is one symbol in one upgrade.
  The parameter rows of a symbol share its label, so counting rows
  counts one observation several times (F12).

  Intervals are 95%, from 2,000 resamples of whole upgrades (version
  pairs), not of rows or changes. The changes in one release tend to be
  used or ignored together, so the upgrade, not the change, is the unit
  that can be treated as independent. An interval that resampled changes
  would be narrower than the data can support.

  Both labels: label_alias, which ships (§24), and label, which §19 and
  §20 used, so each old number has a like-for-like successor. The rows
  are the dev rows only: the holdout stays sealed for final_eval.py.

THE WORDING RULES (NOTES §26.1), fixed before the run

  1. Deprecation. If the interval for (deprecated minus the rest)
     contains 0, the finding is "no detectable difference", with the
     range of differences the data cannot rule out. If it excludes 0,
     it says which way and by how much, with the interval.
  2. The middle peak ("moderate churn raises the rate"). Claimed only if
     1-5 or 6-20 is above the 0 bucket with an interval that excludes 0.
  3. The collapse ("modules with 21+ earlier breaks are almost never
     used"). Claimed only if 21+ is below all the other changes with an
     interval that excludes 0.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.holdout import GROUP, assert_no_holdout  # noqa: E402

DATA = pathlib.Path("data")
ART = pathlib.Path("artifacts")
FEATURES = DATA / "features.csv"
OUT = DATA / "item4_findings.csv"
LABELS = {"label_alias": "ships", "label": "the label §19 and §20 used"}
DRAWS = 2000
SEED = 0
BUCKETS = ["0", "1-5", "6-20", "21+"]


def changes(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    """One row per change (a symbol in one upgrade), and how many changes
    had rows that disagreed on any of the three columns used here. Those
    are counted by their largest value; on clean data there are none."""
    g = df.groupby(GROUP + ["symbol"], sort=False, dropna=False)
    cols = {label: "positive", "was_deprecated_before": "deprecated",
            "prior_breaks_in_module": "prior"}
    out = g[list(cols)].max().rename(columns=cols).reset_index()
    mixed = int((g[list(cols)].nunique() > 1).any(axis=1).sum())
    out["positive"] = out["positive"].astype(int)
    out["deprecated"] = out["deprecated"].astype(int)
    return out, mixed


def bucket(prior: pd.Series) -> pd.Series:
    """Earlier breaks in the module: 0, 1-5, 6-20, 21+ (§19.2's buckets)."""
    p = pd.to_numeric(prior, errors="coerce").fillna(0)
    return pd.cut(p, [-np.inf, 0, 5, 20, np.inf], labels=BUCKETS).astype(str)


def resample(ch: pd.DataFrame, group: pd.Series, cats: list,
             draws: int = DRAWS, seed: int = SEED):
    """Changes and positives per category, and each category's rate in
    `draws` resamples of whole upgrades. Returns (n, k, rates), where
    rates has one row per resample; a resample with no change in some
    category gives NaN there."""
    pair = ch[GROUP].astype(str).agg("\x1f".join, axis=1)
    codes, uniq = pd.factorize(pair)
    col = group.map({c: i for i, c in enumerate(cats)}).to_numpy()
    n = np.zeros((len(uniq), len(cats)))
    k = np.zeros((len(uniq), len(cats)))
    np.add.at(n, (codes, col), 1)
    np.add.at(k, (codes, col), ch["positive"].to_numpy())
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)),
                        size=draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = (w @ k) / (w @ n)
    return n.sum(0), k.sum(0), rates, w @ n, w @ k


def interval(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5]) if len(x) else (np.nan, np.nan)
    return float(lo), float(hi)


def pts(x: float) -> str:
    """A difference of rates, in percentage points."""
    return f"{x * 100:+.2f}"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def deprecation_verdict(d: float, lo: float, hi: float) -> str:
    """Rule 1."""
    if lo <= 0 <= hi:
        return ("no detectable difference. The data are consistent with "
                f"anything from {pts(lo)} to {pts(hi)} points.")
    way = "less" if hi < 0 else "more"
    return (f"deprecated changes are used {way} often: {pts(d)} points "
            f"({pts(lo)} to {pts(hi)}).")


def peak_verdict(lows: dict[str, float]) -> tuple[bool, str]:
    """Rule 2: lows holds the lower end of (bucket minus the 0 bucket)
    for 1-5 and 6-20."""
    clear = [b for b in ("1-5", "6-20") if lows[b] > 0]
    if clear:
        return True, (f"supported: {' and '.join(clear)} above the 0 bucket, "
                      "intervals excluding 0.")
    return False, ("NOT supported: neither 1-5 nor 6-20 is clearly above "
                   "the 0 bucket.")


def collapse_verdict(d: float, lo: float, hi: float) -> tuple[bool, str]:
    """Rule 3: (21+ minus all other changes) and its interval."""
    if hi < 0:
        return True, (f"supported: {pts(d)} points against all other "
                      f"changes ({pts(lo)} to {pts(hi)}).")
    return False, (f"NOT supported: {pts(d)} points against all other "
                   f"changes ({pts(lo)} to {pts(hi)}), an interval that "
                   "includes 0.")


def measure(df: pd.DataFrame, label: str) -> tuple[list[str], list[dict]]:
    """Both findings for one label: the printed lines and the saved rows."""
    ch, mixed = changes(df, label)
    lines, rows = [], []
    if mixed:
        lines.append(f"  ** {mixed:,} changes had rows that disagreed; each "
                     "is counted by its largest value.")

    # 1. Deprecated before removal.
    dep = ch["deprecated"].map({1: "deprecated", 0: "not deprecated"})
    cats = ["deprecated", "not deprecated"]
    n, k, r, _, _ = resample(ch, dep, cats)
    d = r[:, 0] - r[:, 1]
    dlo, dhi = interval(d)
    diff = k[0] / n[0] - k[1] / n[1] if n.all() else float("nan")
    lines.append("  1. DEPRECATED BEFORE IT WAS REMOVED")
    lines.append(f"     {'':<16}{'changes':>9}{'used':>7}{'rate':>8}"
                 f"{'95% interval':>21}")
    for i, c in enumerate(cats):
        lo, hi = interval(r[:, i])
        rate = k[i] / n[i] if n[i] else float("nan")
        lines.append(f"     {c:<16}{int(n[i]):>9,}{int(k[i]):>7,}"
                     f"{pct(rate):>8}{pct(lo) + ' to ' + pct(hi):>21}")
        rows.append({"label": label, "finding": "deprecation", "group": c,
                     "changes": int(n[i]), "used": int(k[i]), "rate": rate,
                     "lo": lo, "hi": hi})
    lines.append(f"     difference {pts(diff)} points, interval {pts(dlo)} "
                 f"to {pts(dhi)}")
    by_row = df.groupby("was_deprecated_before")[label].mean()
    lines.append(f"     counted per row, as §20.2 did: {pct(by_row.get(1, np.nan))}"
                 f" against {pct(by_row.get(0, np.nan))}")
    lines.append(f"     -> {deprecation_verdict(diff, dlo, dhi)}")
    rows.append({"label": label, "finding": "deprecation",
                 "group": "difference", "rate": diff, "lo": dlo, "hi": dhi})

    # 2. Earlier breaks in the same module.
    b = bucket(ch["prior"])
    n, k, r, wn, wk = resample(ch, b, BUCKETS)
    lines.append("\n  2. EARLIER BREAKS IN THE SAME MODULE")
    lines.append(f"     {'prior breaks':<14}{'changes':>9}{'used':>7}"
                 f"{'rate':>8}{'95% interval':>21}   against the 0 bucket")
    row_rate = df.groupby(bucket(df["prior_breaks_in_module"]))[label].mean()
    lows = {}
    for i, c in enumerate(BUCKETS):
        lo, hi = interval(r[:, i])
        rate = k[i] / n[i] if n[i] else float("nan")
        vs = ""
        entry = {"label": label, "finding": "prior_breaks", "group": c,
                 "changes": int(n[i]), "used": int(k[i]), "rate": rate,
                 "lo": lo, "hi": hi, "rate_per_row": row_rate.get(c)}
        if i:
            dd = r[:, i] - r[:, 0]
            dlo_, dhi_ = interval(dd)
            ratio = r[:, i] / r[:, 0]
            rlo, rhi = interval(ratio)
            base = k[0] / n[0] if n[0] else float("nan")
            lows[c] = dlo_
            vs = (f"   {pts(rate - base)} pts ({pts(dlo_)} to {pts(dhi_)}),"
                  f" {rate / base if base else float('nan'):.2f}x "
                  f"({rlo:.2f} to {rhi:.2f})")
            entry.update(diff=rate - base, diff_lo=dlo_, diff_hi=dhi_,
                         ratio=rate / base if base else np.nan,
                         ratio_lo=rlo, ratio_hi=rhi)
        lines.append(f"     {c:<14}{int(n[i]):>9,}{int(k[i]):>7,}"
                     f"{pct(rate):>8}{pct(lo) + ' to ' + pct(hi):>21}{vs}")
        rows.append(entry)
    rest_n, rest_k = wn[:, :3].sum(1), wk[:, :3].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dc = r[:, 3] - rest_k / rest_n
    clo, chi = interval(dc)
    d21 = k[3] / n[3] - k[:3].sum() / n[:3].sum() if n[3] else float("nan")
    lines.append("     counted per row, as §20.2 did: "
                 + ", ".join(f"{c} {pct(row_rate.get(c, np.nan))}"
                             for c in BUCKETS))
    _, peak = peak_verdict(lows)
    _, collapse = collapse_verdict(d21, clo, chi)
    lines.append(f"     -> the middle peak: {peak}")
    lines.append(f"     -> the collapse at 21+: {collapse}")
    rows.append({"label": label, "finding": "prior_breaks",
                 "group": "21+ minus the rest", "rate": d21, "lo": clo,
                 "hi": chi})
    return lines, rows


def gain_shares() -> list[str]:
    imp, met = ART / "importance.csv", ART / "metrics.json"
    if not imp.exists():
        return [f"  {imp} not found, so no gain shares. Run "
                "ml/model/train.py to write it."]
    t = pd.read_csv(imp)
    version = (json.loads(met.read_text()).get("version", "unknown")
               if met.exists() else "unknown")
    t["share"] = t["gain"] / max(t["gain"].sum(), 1e-9)
    t["rank"] = t["gain"].rank(ascending=False, method="min").astype(int)
    lines = [f"  from {imp}, model {version}"]
    if version != "lambdarank-label_alias":
        lines.append("  ** that is not the shipped model "
                     "(lambdarank-label_alias). Re-run ml/model/train.py.")
    for f in ("was_deprecated_before", "prior_breaks_in_module"):
        row = t[t["feature"] == f]
        if row.empty:
            lines.append(f"  {f:<24} not in the table")
            continue
        lines.append(f"  {f:<24}{row['share'].iloc[0]:>7.1%}   rank "
                     f"{row['rank'].iloc[0]} of {len(t)} by gain")
    return lines


def main() -> None:
    if not FEATURES.exists():
        sys.exit(f"{FEATURES} not found. Run ml/features/build.py first.")
    df = pd.read_csv(FEATURES, dtype={c: str for c in GROUP})
    # Descriptive numbers still read labels, so the holdout's labels stay
    # out of them too; a stale features.csv that holds it stops here.
    assert_no_holdout(df, "item4_findings.py")
    n_changes = len(df.drop_duplicates(GROUP + ["symbol"]))
    n_pairs = len(df.drop_duplicates(GROUP))
    print(f"\n{len(df):,} dev rows -> {n_changes:,} changes (a symbol in one "
          f"upgrade), in {n_pairs:,} upgrades")
    print(f"95% intervals from {DRAWS:,} resamples of whole upgrades\n")

    saved = []
    for label, why in LABELS.items():
        print("=" * 74)
        print(f"  {label}   ({why})")
        print("=" * 74)
        lines, rows = measure(df, label)
        print("\n".join(lines) + "\n")
        saved += rows

    print("=" * 74)
    print("  GAIN SHARES IN THE SHIPPED MODEL")
    print("=" * 74)
    print("\n".join(gain_shares()))

    pd.DataFrame(saved).to_csv(OUT, index=False)
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
