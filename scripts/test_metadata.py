"""
F39: is the metadata rule the one NOTES §32 fixed, does build.py drop
those rows where F1 drops version strings, and does
scripts/metadata_count.py count what it says it counts?

    python scripts/test_metadata.py

No network and no real data; about fifteen seconds. It adds metadata rows
to the labelled fixture test_holdout.py builds, runs the real build.py on
it in a temp directory, then the count, and checks both against counts
made here. Nothing in your data/ is read or touched.

Four cases:

  1. The rule: the 15 names §32.1 fixed and the two §32.2 added after the
     count, matched on a changed value only. Not the same names removed,
     not F1's version strings, not other dunders, not a plain `copyright`,
     and not a metadata name in the middle of a path.
  2. build.py drops them (step 2): none reaches features.csv or
     holdout.csv, it says how many went and how many upgrades they
     emptied, a removal and an unlisted dunder stay, and release_size and
     package_churn count only the rows left. The count, run on what
     build.py wrote, then finds none.
  3. The count, on files built as they were before step 2: every total it
     prints matches one made here; the upgrades it says F39 empties are
     exactly those with nothing else in them; a removal is reported as
     staying; a dunder not on the list appears in section 6, and that
     table is the same whatever its rows' labels; nothing in data/ changes.
  4. What the count refuses: a features.csv carrying a holdout row, and
     files built before F1.
"""

import hashlib
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (METADATA_LEAVES, VERSION_LEAVES,  # noqa: E402
                               add_features, drop_version_strings,
                               metadata_strings, temporal_split)
from ml.holdout import GROUP, HOLDOUT_START, split_off  # noqa: E402

# The names NOTES §32.1 fixed on 9 Oct before the count, and the two §32.2
# added after it, written out again here so the rule cannot change through
# an edit to build.py alone.
FIXED = {"__author__", "__credits__", "__date__", "__title__", "__summary__",
         "__uri__", "__email__", "__license__", "__copyright__",
         "__description__", "__url__", "__build__", "__author_email__",
         "__maintainer__", "__status__"}
ADDED = {"__version_info__", "__version_time__"}
NAMES = FIXED | ADDED
VALUE = "ATTRIBUTE_CHANGED_VALUE"
LABELS = ("label", "label_scoped", "label_alias")

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load("metadata_count")


def frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "kind": k} for s, k in rows])


def case_rule() -> None:
    print("\n1. THE RULE, AS NOTES §32 FIXED IT")
    check("METADATA_LEAVES is §32.1's 15 names and §32.2's two",
          sorted(METADATA_LEAVES), sorted(NAMES))
    check("none of them is one of F1's version names",
          sorted(METADATA_LEAVES & VERSION_LEAVES), [])
    every = frame([(f"pkg.__about__.{n}", VALUE) for n in sorted(NAMES)])
    check(f"each of the {len(NAMES)} is matched when its value changed",
          int(metadata_strings(every).sum()), len(NAMES))
    other = frame([(f"pkg.{n}", k) for n in sorted(NAMES)
                   for k in ("OBJECT_REMOVED", "OBJECT_CHANGED_KIND")])
    check("the same names removed, or changed in kind, are not: those stay",
          int(metadata_strings(other).sum()), 0)
    near = frame([("pkg.__version__", VALUE), ("pkg.VERSION", VALUE),
                  ("pkg.__cake__", VALUE), ("pkg.core.Thing.__init__", VALUE),
                  ("pkg.__all__", VALUE), ("pkg.Thing.__slots__", VALUE),
                  ("pkg.Thing.__doc__", VALUE), ("pkg.copyright", VALUE),
                  ("pkg.author", VALUE), ("pkg.__copyright__.year", VALUE),
                  ("pkg.__about__", VALUE)])
    check("not a version string, another dunder, a plain name, or a "
          "metadata name inside a path",
          metadata_strings(near).tolist(), [False] * len(near))


def base_fixture() -> pd.DataFrame:
    """test_holdout.py's labelled.csv: 60 packages, 7 upgrades each, some
    in the holdout. test_features.py builds on the same one."""
    return load("test_holdout").make_labelled()


def added(sel: pd.DataFrame, leaf: str, kind: str, used: int) -> pd.DataFrame:
    """One new row per upgrade in `sel`: pkg.__about__.<leaf>, changed by
    `kind`, used or not under every label."""
    r = sel.copy()
    r["symbol"] = r["package"] + ".__about__." + leaf
    r["kind"], r["sub_target"] = kind, ""
    for lab in LABELS:
        r[lab] = used
    r["user_count"] = 3 if used else 0
    r["is_private"], r["is_dunder"] = False, True
    return r


def with_metadata(base: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """A changed __copyright__ (unused) in every third upgrade; a used
    __author__ value change in two dev upgrades; a removed __author__ in
    one, which must stay; a changed __version_info__, one of the names
    added after the count; a used __title__ change in one holdout upgrade,
    which must never count as used; a changed __cake__, not on the list,
    in two; two upgrades that are nothing but metadata, one in dev and
    one in the holdout; and one that is nothing but a version string, which
    F1 empties first. Returns the frame and the keys of the two."""
    pairs = base.drop_duplicates(GROUP).reset_index(drop=True)
    late = pd.to_datetime(pairs["released_at"]) >= HOLDOUT_START
    dev, held = pairs[~late], pairs[late]
    only = pairs.drop_duplicates("package").head(2).copy()
    only["version_from"], only["version_to"] = "9.9.9", "9.9.10"
    only["released_at"] = [
        (HOLDOUT_START + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for d in (-40, 6)]
    keys = set(only[GROUP].astype(str).itertuples(index=False, name=None))
    bump = pairs.drop_duplicates("package").iloc[[2]].copy()
    bump["version_from"], bump["version_to"] = "9.9.9", "9.9.10"
    bump["released_at"] = (HOLDOUT_START
                           - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    df = pd.concat([
        base,
        added(pairs.iloc[::3], "__copyright__", VALUE, 0),
        added(dev.iloc[[1, 4]], "__author__", VALUE, 1),
        added(dev.iloc[[7]], "__author__", "OBJECT_REMOVED", 0),
        added(dev.iloc[[10]], "__version_info__", VALUE, 0),
        added(held.iloc[[0]], "__title__", VALUE, 1),
        added(pairs.iloc[[2, 5]], "__cake__", VALUE, 1),
        added(only, "__copyright__", VALUE, 0),
        added(only, "__license__", VALUE, 0),
        added(bump, "__version__", VALUE, 1),
    ], ignore_index=True)
    return df, keys


def meta(df: pd.DataFrame) -> pd.Series:
    """The rule, applied here without build.py's code."""
    leaf = df["symbol"].astype(str).str.rsplit(".", n=1).str[-1]
    return leaf.isin(NAMES) & df["kind"].astype(str).eq(VALUE)


def gates_here(df: pd.DataFrame, label: str) -> tuple[int, int, int]:
    p = df.groupby(GROUP)[label].agg(["size", "sum"])
    has = p["sum"] > 0
    return (int(p["sum"].sum()), int((has & (p["size"] > 10)).sum()),
            int((has & (p["size"] > 20)).sum()))


def stories_here(dev: pd.DataFrame) -> int:
    t = dev[(dev["split"] == "test") & (dev["is_private"].astype(int) == 0)]
    g = t.groupby(GROUP)["label_alias"].agg(["size", "sum"])
    return int(((g["sum"] > 0) & (g["size"] >= 20)).sum())


def digest(folder: pathlib.Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


def run(tmp: pathlib.Path, script: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(ROOT / script)], cwd=tmp,
                       capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def number(pattern: str, text: str) -> tuple | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return tuple(int(g.replace(",", "")) for g in m.groups())


def block(out: str, name: str) -> str:
    """Section 1's lines for one name: its line and the indented ones
    under it."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(name + " "):
            j = i + 1
            while j < len(lines) and lines[j].startswith("        "):
                j += 1
            return "\n".join(lines[i:j])
    return ""


def read_built(data: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = {c: str for c in GROUP}
    return (pd.read_csv(data / "features.csv", dtype=keys),
            pd.read_csv(data / "holdout.csv", dtype=keys))


def case_build(tmp: pathlib.Path, df: pd.DataFrame, only: set) -> None:
    print("\n2. build.py DROPS THEM, WHERE F1 DROPS VERSION STRINGS")
    data = tmp / "data"
    data.mkdir()
    df.to_csv(data / "labelled.csv", index=False)
    code, out = run(tmp, "ml/features/build.py")
    check("build.py exits cleanly", code, 0)
    if code:
        print(out[-2000:])
        return
    n = int(meta(df).sum())
    check(f"it reports dropping all {n} metadata rows and the 2 upgrades "
          "they emptied",
          (f"F39: {n:,} package-metadata rows dropped" in out,
           "2 more upgrades held nothing else" in out), (True, True))
    dev, held = read_built(data)
    built = pd.concat([dev, held], ignore_index=True)
    check("no metadata row in features.csv or holdout.csv",
          int(meta(built).sum()), 0)
    versions = int(df["symbol"].str.rsplit(".", n=1).str[-1]
                   .isin(VERSION_LEAVES).sum())
    check("every other row is still there, once", len(built),
          len(df) - n - versions)
    left = set(built[GROUP].itertuples(index=False, name=None))
    check("the upgrades that were only metadata are gone", only & left,
          set())
    leaf = built["symbol"].str.rsplit(".", n=1).str[-1]
    check("a removed __author__ and an unlisted __cake__ stay",
          (int((leaf.eq("__author__") & built["kind"].eq("OBJECT_REMOVED"))
               .sum()), int(leaf.eq("__cake__").sum())), (1, 2))
    size = built.groupby(GROUP)["symbol"].transform("size")
    check("release_size counts only the rows left, so the drop came first",
          bool((built["release_size"] == size).all()), True)
    when = pd.to_datetime(built["released_at"])
    churn = (when.groupby(built["package"]).rank(method="min").sub(1)
             .astype(int))
    check("package_churn counts only the rows left, too",
          int((built["package_churn"] != churn).sum()), 0)

    code, out = run(tmp, "scripts/metadata_count.py")
    check("the count, on what build.py wrote, finds none",
          (code, "F39 finds no rows" in out,
           number(r"dev\s+([\d,]+) of [\d,]+ rows", out)), (0, True, (0,)))


def build_before_step2(df: pd.DataFrame, data: pathlib.Path) -> None:
    """features.csv and holdout.csv as build.py wrote them before step 2:
    version strings out and metadata rows in. main()'s steps, in its
    order, from the CSV it reads, minus the F39 drop."""
    df.to_csv(data / "labelled.csv", index=False)
    raw = pd.read_csv(data / "labelled.csv")
    dev, held = split_off(add_features(drop_version_strings(raw)))
    temporal_split(dev).to_csv(data / "features.csv", index=False)
    held.assign(split="holdout").to_csv(data / "holdout.csv", index=False)


def case_count(tmp: pathlib.Path, df: pd.DataFrame, only: set) -> None:
    print("\n3. THE COUNT, ON FILES BUILT BEFORE STEP 2")
    data = tmp / "before"
    (data / "data").mkdir(parents=True)
    build_before_step2(df, data / "data")
    dev, held = read_built(data / "data")
    md, mh = meta(dev), meta(held)
    check("the files hold every metadata row", int(md.sum() + mh.sum()),
          int(meta(df).sum()))

    before = digest(data / "data")
    code, out = run(data, "scripts/metadata_count.py")
    check("metadata_count.py runs", code, 0)
    if code:
        print(out[-2000:])
        return
    check("and writes nothing: every file in data/ is byte for byte the same",
          digest(data / "data"), before)

    used = int(dev.loc[md, "label_alias"].sum())
    check("section 2, dev: rows taken out, of all, and the used ones",
          number(r"dev\s+([\d,]+) of ([\d,]+) rows .*?; ([\d,]+) of the "
                 r"([\d,]+) used rows", out),
          (int(md.sum()), len(dev), used, int(dev["label_alias"].sum())))
    check("section 2, holdout: rows taken out, of all",
          number(r"holdout\s+([\d,]+) of ([\d,]+) rows", out),
          (int(mh.sum()), len(held)))
    empty = {k for k, g in pd.concat([dev, held]).groupby(GROUP)
             if meta(g).all()}
    check("the upgrades it empties are the two with nothing else in them",
          (empty == only,
           number(r"nothing else in them: dev ([\d,]+), holdout ([\d,]+)",
                  out)), (True, (1, 1)))

    author = block(out, "__author__")
    check("__author__: two dev rows, both used; its removal stays",
          (bool(re.search(r"dev\s+2 \(2 used\)", author)),
           "stays: OBJECT_REMOVED 1" in author), (True, True))
    title = block(out, "__title__")
    check("__title__: used, but in the holdout, so never counted as used",
          bool(re.search(r"dev\s+0 \(0 used\)\s+holdout\s+1\b", title)),
          True)
    lic = block(out, "__license__")
    check("__license__: one row in dev, one in the holdout",
          bool(re.search(r"dev\s+1 \(0 used\)\s+holdout\s+1\b", lic)), True)
    info = block(out, "__version_info__")
    check("__version_info__, added after the count, is counted with them",
          bool(re.search(r"dev\s+1 \(0 used\)\s+holdout\s+0\b", info)), True)
    cop = block(out, "__copyright__")
    want = (int((md & dev["symbol"].str.endswith("__copyright__")).sum()),
            int((mh & held["symbol"].str.endswith("__copyright__")).sum()))
    check("__copyright__: its dev and holdout rows",
          number(r"dev\s+([\d,]+) \(0 used\)\s+holdout\s+([\d,]+)", cop),
          want)
    quiet = re.search(r"no change at all in this data: (.*?)\n\n", out, re.S)
    check("a name nothing changed is listed as such",
          bool(quiet and "__credits__" in quiet.group(1)), True)

    test = dev["split"] == "test"
    check("section 3: rows in the test half, and the used ones",
          number(r"test half: ([\d,]+) rows, ([\d,]+) used", out),
          (int((md & test).sum()),
           int(dev.loc[md & test, "label_alias"].sum())))
    check("section 3: story candidates before and after",
          number(r"story candidates \(§31\.1\): (\d+) now, (\d+) after F39",
                 out),
          (stories_here(dev), stories_here(dev[~md])))

    for lab in ("label_alias", "label"):
        b, a = gates_here(held, lab), gates_here(held[~mh], lab)
        check(f"section 5: the holdout's gates under {lab}, before and after",
              number(lab + r"\s+now (\d+)/(\d+)/(\d+), after F39 "
                     r"(\d+)/(\d+)/(\d+)", out), b + a)

    sec6 = out.split("6. FOR THE RECORD")[-1].split("This script")[0]
    cake = re.search(r"^\s+__cake__\s+(\d+)\s+e\.g\. (\S+)", sec6, re.M)
    check("section 6 lists __cake__, not on the list, with its two rows, "
          "and not __version_info__, which now is",
          (bool(cake), cake and int(cake.group(1)),
           "__version_info__" in sec6), (True, 2, False))
    flipped = [x.assign(**{lab: 1 - x[lab] for lab in LABELS if lab in x})
               for x in (dev, held)]
    check("and section 6's table is the same whatever the labels say",
          M.other_dunders(dev, held).equals(M.other_dunders(*flipped)), True)
    check("section 6 prints no label count", "used" in sec6, False)


def case_refusals(tmp: pathlib.Path) -> None:
    print("\n4. WHAT THE COUNT REFUSES")
    data = tmp / "before"
    feats = data / "data" / "features.csv"
    if not feats.exists():
        check("the files were built (case 3)", False, True)
        return
    keep = feats.read_bytes()
    dev = pd.read_csv(feats, dtype={c: str for c in GROUP})

    day = (HOLDOUT_START + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    late = dev.head(1).assign(released_at=day)
    pd.concat([dev, late]).to_csv(feats, index=False)
    code, out = run(data, "scripts/metadata_count.py")
    check("a features.csv carrying a holdout row is refused",
          (code != 0, "HOLDOUT ROWS" in out), (True, True))

    version = dev.head(1).assign(
        symbol=dev["package"].iloc[0] + ".__version__", kind=VALUE)
    pd.concat([dev, version]).to_csv(feats, index=False)
    code, out = run(data, "scripts/metadata_count.py")
    check("files that still hold a version string (built before F1) are "
          "refused", (code != 0, "before F1" in out), (True, True))
    feats.write_bytes(keep)


def main() -> None:
    case_rule()
    df, only = with_metadata(base_fixture())
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-metadata-"))
    try:
        case_build(tmp, df, only)
        case_count(tmp, df, only)
        case_refusals(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. build.py drops what NOTES §32 says, before any")
    print("feature is computed, and the count says what the data held.")


if __name__ == "__main__":
    main()
