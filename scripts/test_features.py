"""
Item 2 of the fix list: version strings out (F1), package_churn past-only
(F5). Are both really done?

    python scripts/test_features.py

No network and no real data; about ten seconds. It adds version-string
rows to the labelled fixture test_holdout.py builds, runs the real
build.py on it in a temp directory, and checks what came out.

WHY THIS FILE EXISTS. Both fixes change what every model trains on, and
both can look done while being half-done:

  F1  a version string dropped AFTER add_features still counts in its
      upgrade's release_size and its package's churn, so the model keeps
      seeing it through two features after it was "removed".
  F5  a churn count that includes later releases looks exactly like one
      that does not. The only way to tell is to add rows from the future
      and watch whether earlier rows change.

Four cases:

  1. build.py writes no version-string row anywhere, says how many it
     dropped, and upgrades that were only a version bump are gone.
     scripts/holdout_boundary.py still runs on what it wrote.
  2. release_size counts only the rows that are left, so the drop came
     before the features. is_version_string is no longer a feature or a
     column, and every ablation group names only real features.
  3. package_churn is how many of the package's rows were released
     strictly earlier: 0 for its first release, never its own.
  4. Adding later releases (the holdout, or the next ingest) changes no
     earlier row's churn. The whole-file count it replaced fails that on
     the same data, and that is checked too, so the case cannot pass by
     the fixture having nothing to find.
"""

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (BOOLEAN, CATEGORICAL, NUMERIC,  # noqa: E402
                               add_features, drop_version_strings,
                               version_strings)
from ml.holdout import GROUP, HOLDOUT_START, holdout_mask  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def base_fixture() -> pd.DataFrame:
    """test_holdout.py's labelled.csv, so both files test the same shape of
    data: 60 packages, 7 upgrades each, some of them in the holdout."""
    spec = importlib.util.spec_from_file_location(
        "test_holdout", ROOT / "scripts" / "test_holdout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_labelled()


def with_version_strings(base: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Every upgrade gains a positive pkg.__version__ row, and three new
    upgrades are nothing but a version bump (one in dev, two in the
    holdout). Returns the frame and the keys of those three."""
    pairs = base.drop_duplicates(GROUP)
    bumps = pairs.assign(symbol=pairs["package"] + ".__version__",
                         kind="ATTRIBUTE_CHANGED_VALUE", sub_target="",
                         label=1, label_scoped=1, label_alias=1,
                         user_count=5)
    only = pairs.drop_duplicates("package").head(3).copy()
    only["version_from"], only["version_to"] = "9.9.9", "9.9.10"
    start = HOLDOUT_START
    only["released_at"] = [(start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
                           for d in (-30, 3, 9)]
    only = only.assign(symbol=only["package"] + ".VERSION",
                       kind="ATTRIBUTE_CHANGED_VALUE", sub_target="",
                       label=1, label_scoped=1, label_alias=1, user_count=5)
    keys = set(only[GROUP].astype(str).itertuples(index=False, name=None))
    df = pd.concat([base, bumps, only], ignore_index=True)
    return df, keys


def case_build(tmp: pathlib.Path, df: pd.DataFrame, only: set) -> None:
    print("\n1. build.py DROPS EVERY VERSION STRING, AND SAYS SO")
    (tmp / "data").mkdir()
    df.to_csv(tmp / "data" / "labelled.csv", index=False)
    p = subprocess.run([sys.executable, str(ROOT / "ml/features/build.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=600)
    check("build.py exits cleanly", p.returncode, 0)
    if p.returncode:
        print((p.stdout + p.stderr)[-2000:])
        return
    n_vs = int(version_strings(df).sum())
    check(f"it reports dropping all {n_vs} version-string rows and the 3 "
          "upgrades they emptied",
          (f"F1: {n_vs:,} version-string rows dropped" in p.stdout,
           "3 upgrades held nothing else" in p.stdout), (True, True))
    keys = {c: str for c in GROUP}
    out = pd.concat([pd.read_csv(tmp / "data" / "features.csv", dtype=keys),
                     pd.read_csv(tmp / "data" / "holdout.csv", dtype=keys)],
                    ignore_index=True)
    check("no version-string row in features.csv or holdout.csv",
          int(version_strings(out).sum()), 0)
    check("every other row is still there, once",
          len(out), len(df) - n_vs)
    left = set(out[GROUP].itertuples(index=False, name=None))
    check("the upgrades that were only a version bump are gone",
          only & left, set())
    q = subprocess.run([sys.executable,
                        str(ROOT / "scripts/holdout_boundary.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=600)
    check("holdout_boundary.py still runs, and finds F1 already done",
          (q.returncode, "F1 deletes 0 rows" in q.stdout), (0, True))
    if q.returncode:
        print((q.stdout + q.stderr)[-1500:])

    print("\n2. THE DROP CAME BEFORE THE FEATURES")
    size = out.groupby(GROUP)["symbol"].transform("size")
    check("release_size counts only the rows that are left",
          bool((out["release_size"] == size).all()), True)
    features = NUMERIC + BOOLEAN + CATEGORICAL
    check("is_version_string is not a feature and not a column",
          ("is_version_string" in features,
           "is_version_string" in out.columns), (False, False))
    from ml.model import ablate
    groups = {g: getattr(ablate, g) for g in
              ("PATH_SHAPE", "REACHABILITY", "BLAST_RADIUS", "HISTORY",
               "POPULARITY", "PER_CHANGE") if hasattr(ablate, g)}
    unknown = sorted({f for g in groups.values() for f in g} - set(features))
    check(f"every ablation group names only real features "
          f"({len(groups)} groups)", unknown, [])


def small(rows: list[tuple[str, str, str, str, int]]) -> pd.DataFrame:
    """A minimal labelled frame: (package, from, to, date, n rows) each."""
    out = []
    for pkg, vf, vt, when, n in rows:
        for i in range(n):
            out.append({"package": pkg, "version_from": vf, "version_to": vt,
                        "released_at": when, "symbol": f"{pkg}.api.f{vt}_{i}",
                        "kind": "OBJECT_REMOVED", "sub_target": "",
                        "module_depth": 2, "public_depth": 1,
                        "name_length": 4, "package_rank": 1,
                        "inherited_by": 0, "prior_breaks_in_module": 0,
                        "was_deprecated_before": False, "is_private": False,
                        "is_dunder": False, "in_dunder_all": False,
                        "is_top_level": True, "has_export_path": False,
                        "label": 0})
    return pd.DataFrame(out)


def case_past_only() -> None:
    print("\n3. package_churn COUNTS ONLY EARLIER RELEASES")
    df = add_features(small([
        ("a", "1.0", "1.1", "2026-01-10", 3),
        ("a", "1.1", "1.2", "2026-02-10", 2),
        ("a", "1.2", "1.3", "2026-03-10", 4),
        ("b", "2.0", "2.1", "2026-02-10", 5),
    ]))
    got = (df.groupby(["package", "version_to"])["package_churn"]
           .agg(["min", "max"]).apply(tuple, axis=1).to_dict())
    check("a: 0 before its first release, then 3, then 3 + 2; b starts at "
          "0 though a released the same day",
          got, {("a", "1.1"): (0, 0), ("a", "1.2"): (3, 3),
                ("a", "1.3"): (5, 5), ("b", "2.1"): (0, 0)})


def case_no_future(base: pd.DataFrame) -> None:
    print("\n4. LATER RELEASES CHANGE NO EARLIER ROW'S CHURN")
    df = drop_version_strings(base)
    dev = df[~holdout_mask(df)]
    full = add_features(df).loc[dev.index, "package_churn"]
    alone = add_features(dev)["package_churn"]
    check("every dev row's churn is the same with the holdout present or "
          "absent", int((full != alone).sum()), 0)

    # The control. The count F5 replaced, on the same two frames: if this
    # found no difference, case 4 would be passing on a fixture that has no
    # later releases to leak.
    old_full = df.groupby("package")["symbol"].transform("size")[dev.index]
    old_alone = dev.groupby("package")["symbol"].transform("size")
    moved = int((old_full != old_alone).sum())
    check("control: the whole-file count it replaced does change "
          f"({moved:,} dev rows)", moved > 0, True)


def main() -> None:
    base = base_fixture()
    df, only = with_version_strings(base)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-features-"))
    try:
        case_build(tmp, df, only)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    case_past_only()
    case_no_future(df)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. No version string reaches a feature, and no")
    print("row's churn counts a release that came after it.")


if __name__ == "__main__":
    main()
