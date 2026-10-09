"""
F39, step 1: is the metadata rule the one NOTES §32.1 fixed, and does
scripts/metadata_count.py count what it says it counts?

    python scripts/test_metadata.py

No network and no real data; about ten seconds. It adds metadata rows to
the labelled fixture test_holdout.py builds, runs the real build.py on it
in a temp directory, then the count, and checks what the count printed
against counts made here. Nothing in your data/ is read or touched.

Three cases:

  1. The rule: the 15 names NOTES §32.1 lists, matched on a changed value
     only. Not the same names removed, not F1's version strings, not
     other dunders, not a plain `copyright`, and not a metadata name in
     the middle of a path.
  2. The count: every total it prints matches one made here; the
     upgrades it says F39 empties are exactly those with nothing else in
     them; a removal is reported as staying; a dunder not on the list
     appears in section 6, and that table is the same whatever its rows'
     labels; and nothing in data/ changes.
  3. What it refuses: a features.csv carrying a holdout row, and files
     built before F1.
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
                               metadata_strings)
from ml.holdout import GROUP, HOLDOUT_START  # noqa: E402

# The names NOTES §32.1 fixed on 9 Oct, written out again here so the rule
# cannot change through an edit to build.py alone.
FIXED = {"__author__", "__credits__", "__date__", "__title__", "__summary__",
         "__uri__", "__email__", "__license__", "__copyright__",
         "__description__", "__url__", "__build__", "__author_email__",
         "__maintainer__", "__status__"}
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
    print("\n1. THE RULE, AS NOTES §32.1 FIXED IT")
    check("METADATA_LEAVES is the 15 names NOTES §32.1 lists",
          sorted(METADATA_LEAVES), sorted(FIXED))
    check("none of them is one of F1's version names",
          sorted(METADATA_LEAVES & VERSION_LEAVES), [])
    every = frame([(f"pkg.__about__.{n}", VALUE) for n in sorted(FIXED)])
    check("each of the 15 is matched when its value changed",
          int(metadata_strings(every).sum()), len(FIXED))
    other = frame([(f"pkg.{n}", k) for n in sorted(FIXED)
                   for k in ("OBJECT_REMOVED", "OBJECT_CHANGED_KIND")])
    check("the same names removed, or changed in kind, are not: those stay",
          int(metadata_strings(other).sum()), 0)
    near = frame([("pkg.__version__", VALUE), ("pkg.VERSION", VALUE),
                  ("pkg.__cake__", VALUE), ("pkg.core.Thing.__init__", VALUE),
                  ("pkg.__all__", VALUE), ("pkg.copyright", VALUE),
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
    one, which must stay; a used __title__ change in one holdout upgrade,
    which must never count as used; a changed __cake__, not on the list,
    in two; and two upgrades that are nothing but metadata, one in dev and
    one in the holdout. Returns the frame and the keys of those two."""
    pairs = base.drop_duplicates(GROUP).reset_index(drop=True)
    late = pd.to_datetime(pairs["released_at"]) >= HOLDOUT_START
    dev, held = pairs[~late], pairs[late]
    only = pairs.drop_duplicates("package").head(2).copy()
    only["version_from"], only["version_to"] = "9.9.9", "9.9.10"
    only["released_at"] = [
        (HOLDOUT_START + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for d in (-40, 6)]
    keys = set(only[GROUP].astype(str).itertuples(index=False, name=None))
    df = pd.concat([
        base,
        added(pairs.iloc[::3], "__copyright__", VALUE, 0),
        added(dev.iloc[[1, 4]], "__author__", VALUE, 1),
        added(dev.iloc[[7]], "__author__", "OBJECT_REMOVED", 0),
        added(held.iloc[[0]], "__title__", VALUE, 1),
        added(pairs.iloc[[2, 5]], "__cake__", VALUE, 1),
        added(only, "__copyright__", VALUE, 0),
        added(only, "__license__", VALUE, 0),
    ], ignore_index=True)
    return df, keys


def meta(df: pd.DataFrame) -> pd.Series:
    """The rule, applied here without build.py's code."""
    leaf = df["symbol"].astype(str).str.rsplit(".", n=1).str[-1]
    return leaf.isin(FIXED) & df["kind"].astype(str).eq(VALUE)


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


def run_count(tmp: pathlib.Path) -> tuple[int, str]:
    script = ROOT / "scripts" / "metadata_count.py"
    p = subprocess.run([sys.executable, str(script)], cwd=tmp,
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


def case_count(tmp: pathlib.Path, df: pd.DataFrame, only: set) -> None:
    print("\n2. THE COUNT, CHECKED AGAINST ONE MADE HERE")
    (tmp / "data").mkdir()
    df.to_csv(tmp / "data" / "labelled.csv", index=False)
    p = subprocess.run([sys.executable, str(ROOT / "ml/features/build.py")],
                       cwd=tmp, capture_output=True, text=True, timeout=600)
    check("build.py builds the fixture", p.returncode, 0)
    if p.returncode:
        print((p.stdout + p.stderr)[-2000:])
        return
    keys = {c: str for c in GROUP}
    dev = pd.read_csv(tmp / "data" / "features.csv", dtype=keys)
    held = pd.read_csv(tmp / "data" / "holdout.csv", dtype=keys)
    md, mh = meta(dev), meta(held)
    check("step 1 changes no build: every metadata row is still there",
          int(md.sum() + mh.sum()), int(meta(df).sum()))

    before = digest(tmp / "data")
    code, out = run_count(tmp)
    check("metadata_count.py runs", code, 0)
    if code:
        print(out[-2000:])
        return
    check("and writes nothing: every file in data/ is byte for byte the same",
          digest(tmp / "data"), before)

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

    cake = re.search(r"^\s+__cake__\s+(\d+)\s+e\.g\. (\S+)", out, re.M)
    check("section 6 lists __cake__, not on the list, with its two rows",
          (bool(cake), cake and int(cake.group(1))), (True, 2))
    flipped = [x.assign(**{lab: 1 - x[lab] for lab in LABELS if lab in x})
               for x in (dev, held)]
    check("and section 6's table is the same whatever the labels say",
          M.other_dunders(dev, held).equals(M.other_dunders(*flipped)), True)
    sec6 = out.split("6. FOR THE RECORD")[-1].split("This script")[0]
    check("section 6 prints no label count", "used" in sec6, False)


def case_refusals(tmp: pathlib.Path) -> None:
    print("\n3. WHAT IT REFUSES")
    feats = tmp / "data" / "features.csv"
    if not feats.exists():
        check("the fixture was built (case 2)", False, True)
        return
    keep = feats.read_bytes()
    dev = pd.read_csv(feats, dtype={c: str for c in GROUP})

    day = (HOLDOUT_START + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    late = dev.head(1).assign(released_at=day)
    pd.concat([dev, late]).to_csv(feats, index=False)
    code, out = run_count(tmp)
    check("a features.csv carrying a holdout row is refused",
          (code != 0, "HOLDOUT ROWS" in out), (True, True))

    version = dev.head(1).assign(
        symbol=dev["package"].iloc[0] + ".__version__", kind=VALUE)
    pd.concat([dev, version]).to_csv(feats, index=False)
    code, out = run_count(tmp)
    check("files that still hold a version string (built before F1) are "
          "refused", (code != 0, "before F1" in out), (True, True))
    feats.write_bytes(keep)


def main() -> None:
    case_rule()
    df, only = with_metadata(base_fixture())
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-metadata-"))
    try:
        case_count(tmp, df, only)
        case_refusals(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The rule is the one NOTES §32.1 fixed, and the")
    print("count says what the data holds without changing any of it.")


if __name__ == "__main__":
    main()
