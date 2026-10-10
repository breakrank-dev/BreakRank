"""
Which half of item 2 moved the lift: F1 (version strings out) or F5
(package_churn counts earlier releases only)?

    python scripts/item2_effects.py
    python scripts/item2_effects.py --label label_alias

Reads  data/labelled.csv, and data/stability_<label>_cv_at.csv if present
Writes data/item2_effects_<label>.csv. Nothing else is touched.

WHY THIS EXISTS. Item 2 put two fixes in at once. On the frozen dev set,
at the seven cut dates of NOTES §22.6, together they lowered the lift
over popularity at six of the seven (§23). The audit had measured each
as [better] on §20's setup. This builds the pipeline four ways from the
same labelled.csv (neither fix, F1 only, F5 only, both) and runs the
same sweep at the same dates on each, so each fix's share of the change
is measured rather than guessed.

IT CHECKS ITSELF. "Neither" has to reproduce §22.6's sweep and "both" the
--at sweep of 26 Sep (data/stability_<label>_cv_at.csv), lift for lift.
If either does not, the variants are not the pipelines they claim to be,
and the table says so above the numbers rather than below them.

Each variant is written to a temporary features file and read back the
way stability.py reads data/features.csv, so the rows reach the sweep
with exactly the types they had in the real runs. The holdout comes off
after the features are computed and before anything is fitted, the same
order as build.py.
"""

import argparse
import pathlib
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import (BOOLEAN, CATEGORICAL, NUMERIC,  # noqa: E402
                               add_features, drop_version_strings,
                               version_strings)
from ml.holdout import assert_no_holdout, drop_holdout  # noqa: E402
from ml.model import stability  # noqa: E402
from ml.model.train import prepare  # noqa: E402

DATA = pathlib.Path("data")
LABELLED = DATA / "labelled.csv"

# The seven cut dates of the last sweep before item 2 (NOTES §22.6).
DATES = ["2025-09-29", "2025-11-28", "2026-01-13", "2026-02-27",
         "2026-03-29", "2026-04-24", "2026-05-18"]
# That sweep's lifts over popularity under `label`, the "neither" check.
BEFORE_LABEL = [3.22, 3.24, 4.18, 2.98, 3.05, 3.31, 2.82]

# The feature list as it stood before F1: is_version_string sat in
# BOOLEAN after is_top_level. Kept in that position because LightGBM
# samples columns by position, so a reordered list is a different model.
OLD_BOOLEAN = (BOOLEAN[:BOOLEAN.index("is_top_level") + 1]
               + ["is_version_string"]
               + BOOLEAN[BOOLEAN.index("is_top_level") + 1:])


def variant(labelled: pd.DataFrame, f1: bool, f5: bool
            ) -> tuple[pd.DataFrame, list[str]]:
    """The dev rows and feature list one version of build.py produced."""
    df = drop_version_strings(labelled) if f1 else labelled
    df = add_features(df)
    if f1:
        feats = NUMERIC + BOOLEAN + CATEGORICAL
    else:
        df["is_version_string"] = version_strings(df).astype(int)
        feats = NUMERIC + OLD_BOOLEAN + CATEGORICAL
    if not f5:
        # The count F5 replaced: every row of the package, on the whole
        # frame, holdout included, exactly as build.py computed it.
        df["package_churn"] = df.groupby("package")["symbol"].transform(
            "size")
    dev = drop_holdout(df)
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        dev.to_csv(tmp.name, index=False)
        dev = prepare(pd.read_csv(tmp.name))
    assert_no_holdout(dev, "item2_effects.py")
    return dev, feats


def reference(label: str) -> dict[str, list[float]]:
    """Lifts the variants must reproduce, where a record of them exists."""
    ref = {}
    if label == "label":
        ref["neither"] = BEFORE_LABEL
    at = DATA / f"stability_{label}_cv_at.csv"
    if at.exists():
        t = pd.read_csv(at)
        if t["cut"].astype(str).tolist() == DATES:
            ref["both"] = [None if s else round(float(v), 2) for s, v in
                           zip(t["skipped"].astype(bool), t["lift_vs_pop"])]
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description="Split item 2's effect "
                                 "between F1 and F5, at fixed dates.")
    ap.add_argument("--label", default="label",
                    choices=["label", "label_scoped", "label_alias"])
    args = ap.parse_args()
    if not LABELLED.exists():
        sys.exit(f"{LABELLED} not found. Run ml/features/labels.py first.")
    labelled = pd.read_csv(LABELLED)

    runs = {"neither": (False, False), "F1 only": (True, False),
            "F5 only": (False, True), "both": (True, True)}
    lifts, medians = {}, {}
    for name, (f1, f5) in runs.items():
        dev, feats = variant(labelled, f1, f5)
        print(f"{name:<8} {len(dev):,} dev rows, {len(feats)} features ...",
              flush=True)
        t = stability.run_label(dev, args.label, feats, "lambdarank", "cv",
                                DATES)
        t = t.set_index("cut").reindex(DATES)
        lift = [None if s else round(float(v), 2) for s, v in
                zip(t["skipped"].astype(bool), t["lift_vs_pop"])]
        lifts[name] = lift
        kept = [v for v in lift if v is not None]
        medians[name] = (pd.Series(kept).median(), min(kept), len(kept))

    print(f"\nLIFT OVER POPULARITY, label={args.label}, at the seven dates "
          "of NOTES §22.6\n")
    ref = reference(args.label)
    for name, want in ref.items():
        same = lifts[name] == want
        verdict = "reproduces" if same else "DOES NOT reproduce"
        print(f"  self-check: '{name}' {verdict} its recorded run")
        if not same:
            print(f"      recorded {want}\n      this run {lifts[name]}")
    if ref and not all(lifts[n] == w for n, w in ref.items()):
        print("  ** A variant does not match the pipeline it stands for. "
              "Do not read\n  ** the table below until that is explained.")
    print()
    head = "".join(f"{n:>10}" for n in runs)
    print(f"  {'cut':<12}{head}")
    for i, d in enumerate(DATES):
        cells = "".join(f"{('-' if lifts[n][i] is None else lifts[n][i]):>10}"
                        for n in runs)
        print(f"  {d:<12}{cells}")
    print(f"  {'median':<12}"
          + "".join(f"{medians[n][0]:>10.2f}" for n in runs))
    print(f"  {'worst':<12}"
          + "".join(f"{medians[n][1]:>10.2f}" for n in runs))
    print(f"  {'cuts':<12}"
          + "".join(f"{medians[n][2]:>10}" for n in runs))

    out = DATA / f"item2_effects_{args.label}.csv"
    pd.DataFrame({"cut": DATES, **lifts}).to_csv(out, index=False)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
