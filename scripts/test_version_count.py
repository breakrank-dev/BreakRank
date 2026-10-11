"""
F1 narrowed (NOTES §33): is the version-string rule the one §33.1 fixed,
does build.py drop exactly those rows, and does scripts/version_count.py
count what it says it counts?

    python scripts/test_version_count.py

No network and no real data; about fifteen seconds. It adds version rows
of every shape to the labelled fixture test_holdout.py builds, runs the
real build.py and the count on it in a temp directory, and checks both
against counts made here. Nothing in your data/ is read or touched.

Four cases:

  1. The rule, written out again here so it cannot change through an edit
     to build.py alone: a changed value only, and not one changed to
     `unset`; __version__, __VERSION__ and __version_tuple__ wherever they
     sit; version, VERSION and version_tuple only where griffe's
     explanation names them bare, which it does only directly in a module,
     read from the raw explanation and from the database's griffe_message
     alike. Not a removal, a submodule, a method, a class's attribute, an
     empty or unreadable explanation, or a frame with no explanation at
     all. And F39's names: a value changed to `unset` stays there too.
  2. build.py drops exactly those rows, before the features. A removed
     __version__, a class's VERSION, a removed submodule `version`, a
     VERSION griffe cannot place, and a __version__ and a __copyright__
     changed to unset all stay; the upgrades that held only a removal or
     a class's VERSION are no longer emptied.
  3. The count, on the same labelled.csv: all four rules' rows, the used
     ones (dev only), the upgrades emptied; what comes back and why; every
     symbol behind the bare names; the all-clear before and after, the
     releases that leave it, and the preset from the most downloaded
     package; what the loader stops writing; the holdout's rows, pairs,
     fingerprint and gates. The fingerprint after is the one build.py's
     holdout.csv has. Nothing in data/ changes.
  4. What the count refuses: no labelled.csv, one without griffe's
     explanation, a build.py whose names changed, and a rule that takes a
     row the old one did not.
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.build import (METADATA_LEAVES,  # noqa: E402
                               VERSION_LEAVES, metadata_strings,
                               version_strings)
from ml.holdout import (GROUP, HOLDOUT_START, fingerprint,  # noqa: E402
                        holdout_mask)

# NOTES §33.1, written out again.
DUNDERS = {"__version__", "__VERSION__", "__version_tuple__"}
BARE = {"version", "VERSION", "version_tuple"}
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


def said(path: str, inner: str, kind: str = VALUE, new: str = "'1.1'") -> str:
    """griffe's explanation as the ingest stores it: the file and line,
    the path inside the module, the sentence for the kind."""
    sentence = {VALUE: f"Attribute value was changed: '1.0' -> {new}",
                "OBJECT_REMOVED": "Public object was removed",
                "PARAMETER_REMOVED": "Parameter was removed"}[kind]
    return f"data/sdists/x/1.0/{path}:7: {inner}: {sentence}"


def case_rule() -> None:
    print("\n1. THE RULE, AS NOTES §33.1 FIXED IT")
    check("the six names are the six F1 always had",
          sorted(VERSION_LEAVES), sorted(DUNDERS | BARE))
    rows = [
        # (symbol, kind, explanation, goes)
        ("pkg.__version__", VALUE, said("pkg/__init__.py", "__version__"),
         True),
        ("pkg.__VERSION__", VALUE, said("pkg/__init__.py", "__VERSION__"),
         True),
        ("pkg._version.__version_tuple__", VALUE,
         said("pkg/_version.py", "__version_tuple__"), True),
        ("pkg.Klass.__version__", VALUE,
         said("pkg/k.py", "Klass.__version__"), True),
        ("pkg.__version__", VALUE, "", True),
        ("pkg._version.version", VALUE, said("pkg/_version.py", "version"),
         True),
        ("pkg._version.version_tuple", VALUE,
         said("pkg/_version.py", "version_tuple"), True),
        ("PIL.Image.VERSION", VALUE, said("PIL/Image.py", "VERSION"), True),
        ("pkg.VERSION", VALUE,
         "VERSION: Attribute value was changed: '1' -> '2'", True),
        ("pkg.sub.version", VALUE,
         "my dir: v1/pkg/sub.py:3: version: Attribute value was changed: "
         "'x: version: Attribute value was changed' -> 'y'", True),
        ("pkg.__version__", VALUE,
         said("pkg/__init__.py", "__version__", new="unset"), False),
        ("pkg.__version__", VALUE,
         "__version__: Attribute value was changed: '1.0' -> unset", False),
        ("pkg._version.version", VALUE,
         said("pkg/_version.py", "version", new="unset"), False),
        ("pkg.__version__", "OBJECT_REMOVED",
         said("pkg/__init__.py", "__version__", "OBJECT_REMOVED"), False),
        ("pkg.version", "OBJECT_REMOVED",
         said("pkg/version.py", "<module>", "OBJECT_REMOVED"), False),
        ("pkg.Thing.version", "PARAMETER_REMOVED",
         said("pkg/t.py", "Thing.version(strict)", "PARAMETER_REMOVED"),
         False),
        ("pkg.Thing.version", "OBJECT_REMOVED",
         said("pkg/t.py", "Thing.version", "OBJECT_REMOVED"), False),
        ("pkg.core.Config.VERSION", VALUE,
         said("pkg/core.py", "Config.VERSION"), False),
        ("pkg.Sub.version", VALUE,
         "p.py:3: Sub.version: Attribute value was changed: 'version: "
         "Attribute value was changed' -> 'y'", False),
        ("pkg.uni.संस्करण.version", VALUE,
         "p.py:3: संस्करण.version: Attribute value was changed: 'x: version: "
         "Attribute value was changed 1.0' -> 'y'", False),
        ("pkg.VERSION", VALUE, "", False),
        ("pkg.VERSION", VALUE, None, False),
        ("pkg.VERSION", VALUE, "something griffe never wrote", False),
        ("pkg.versions", VALUE, said("pkg/__init__.py", "versions"), False),
        ("pkg.version.Version", VALUE, said("pkg/version.py", "Version"),
         False),
    ]
    df = pd.DataFrame([{"symbol": s, "kind": k, "explanation": e}
                       for s, k, e, _ in rows])
    got = version_strings(df).tolist()
    for (s, k, e, want), g in zip(rows, got):
        if g != want:
            print(f"          {s} {k} {e!r}: got {g}, want {want}")
    check("each row goes or stays as §33.1 says", got,
          [w for *_, w in rows])
    no_col = df.drop(columns="explanation")
    check("with no explanation column, only the dunders' value changes go",
          version_strings(no_col).tolist(),
          [k == VALUE and s.rsplit(".", 1)[-1] in DUNDERS
           for s, k, _, _ in rows])
    check("and the answer is plain booleans", str(version_strings(df).dtype),
          "bool")
    meta = pd.DataFrame([
        {"symbol": "pkg.__about__.__copyright__", "kind": VALUE,
         "explanation": said("pkg/__about__.py", "__copyright__")},
        {"symbol": "pkg.__about__.__copyright__", "kind": VALUE,
         "explanation": said("pkg/__about__.py", "__copyright__",
                             new="unset")},
        {"symbol": "pkg.__about__.__copyright__", "kind": "OBJECT_REMOVED",
         "explanation": said("pkg/__about__.py", "__copyright__",
                             "OBJECT_REMOVED")}])
    check("F39: a changed __copyright__ goes; changed to unset, or removed, "
          "it stays", metadata_strings(meta).tolist(), [True, False, False])


def base_fixture() -> pd.DataFrame:
    """test_holdout.py's labelled.csv: 60 packages, 7 upgrades each, some
    in the holdout, no version names among its symbols."""
    return load("test_holdout").make_labelled()


def rows_for(sel: pd.DataFrame, path: str, kind: str, inner: str,
             where: str, used: int, explain: str | None = None,
             new: str = "'1.1'") -> pd.DataFrame:
    """One row per upgrade in `sel`: <package>.<path>, changed by `kind`,
    with griffe's explanation, used or not under every label."""
    r = sel.copy()
    r["symbol"] = r["package"] + "." + path
    r["kind"], r["sub_target"] = kind, ""
    r["explanation"] = (said(where, inner, kind, new) if explain is None
                        else explain)
    for lab in LABELS:
        r[lab] = used
    r["user_count"] = 3 if used else 0
    r["is_private"] = path.split(".")[0] == "_version"
    return r


def fixture() -> tuple[pd.DataFrame, dict, dict]:
    """The base, plus:

      every upgrade   a changed __version__, used (goes, as always)
      3 dev, 1 held   a changed _version.version in a module (goes)
      1 dev           a changed version_tuple in a module (goes)
      1 dev           VERSION changed, explanation as the database keeps it
                      (goes)
      2 dev, 1 held   a removed __version__, used in one dev and the held
                      one (comes back)
      2 dev           a class's VERSION changed (comes back)
      1 dev           a submodule `version` removed (comes back)
      1 dev           VERSION changed, no explanation (comes back)
      1 dev           __version__ changed to unset, used (comes back)
      3 dev           a changed __copyright__ (F39, goes)
      1 dev           __copyright__ changed to unset (comes back from F39)

    and five upgrades with nothing else in them:

      A  dev, most downloaded package   only a changed __version__
      B  holdout                        only a changed _version.version
      C  dev                            only a removed __version__
      D  holdout                        only a class's VERSION changed
      E  dev                            a changed __version__ and
                                        __copyright__, only the two

    and F, a holdout upgrade left with exactly 10 rows once F1 has run,
    one of them used, so the gates' "more than 10" is tested at its edge.

    Returns the frame, the four rules' masks worked out here, and the keys
    of A to E.
    """
    base = base_fixture()
    pairs = base.drop_duplicates(GROUP).reset_index(drop=True)
    late = pd.to_datetime(pairs["released_at"]) >= HOLDOUT_START
    dev, held = pairs[~late].reset_index(drop=True), pairs[late]
    held = held.reset_index(drop=True)
    firsts = pairs.drop_duplicates("package").reset_index(drop=True)

    def only(i: int, days: int) -> pd.DataFrame:
        o = firsts.iloc[[i]].copy()
        o["version_from"], o["version_to"] = "9.9.9", "9.9.10"
        o["released_at"] = (HOLDOUT_START + pd.Timedelta(days=days)
                            ).strftime("%Y-%m-%d")
        return o

    a, b, c, d, e = (only(0, -10), only(1, 5), only(2, -30), only(3, 8),
                     only(4, -50))
    f = base.merge(held.iloc[[2]][GROUP], on=GROUP).head(10).copy()
    f["version_from"], f["version_to"] = "8.8.8", "8.8.9"
    for lab in LABELS:
        f[lab] = [1] + [0] * (len(f) - 1)
    keys = {name: tuple(x[GROUP].astype(str).iloc[0])
            for name, x in zip("ABCDE", (a, b, c, d, e))}
    v, about = "__init__.py", "__about__.py"
    parts = [
        base, f,
        rows_for(f.head(1), "__version__", VALUE, "__version__", v, 1),
        rows_for(pairs, "__version__", VALUE, "__version__", v, 1),
        rows_for(pd.concat([dev.iloc[[0, 5, 9]], held.iloc[[0]]]),
                 "_version.version", VALUE, "version", "_version.py", 0),
        rows_for(dev.iloc[[2]], "_version.version_tuple", VALUE,
                 "version_tuple", "_version.py", 0),
        rows_for(dev.iloc[[3]], "VERSION", VALUE, "", v, 0,
                 explain="VERSION: Attribute value was changed: 1 -> 2"),
        rows_for(dev.iloc[[4]], "__version__", "OBJECT_REMOVED",
                 "__version__", v, 1),
        rows_for(dev.iloc[[6]], "__version__", "OBJECT_REMOVED",
                 "__version__", v, 0),
        rows_for(held.iloc[[1]], "__version__", "OBJECT_REMOVED",
                 "__version__", v, 1),
        rows_for(dev.iloc[[7, 8]], "core.Config.VERSION", VALUE,
                 "Config.VERSION", "core.py", 0),
        rows_for(dev.iloc[[10]], "version", "OBJECT_REMOVED", "<module>",
                 "version.py", 0),
        rows_for(dev.iloc[[11]], "VERSION", VALUE, "", v, 0, explain=""),
        rows_for(dev.iloc[[15]], "__version__", VALUE, "__version__", v, 1,
                 new="unset"),
        rows_for(dev.iloc[[12, 13, 14]], "__about__.__copyright__", VALUE,
                 "__copyright__", about, 0),
        rows_for(dev.iloc[[16]], "__about__.__copyright__", VALUE,
                 "__copyright__", about, 0, new="unset"),
        rows_for(a, "__version__", VALUE, "__version__", v, 1),
        rows_for(b, "_version.version", VALUE, "version", "_version.py", 0),
        rows_for(c, "__version__", "OBJECT_REMOVED", "__version__", v, 0),
        rows_for(d, "core.Config.VERSION", VALUE, "Config.VERSION",
                 "core.py", 0),
        rows_for(e, "__version__", VALUE, "__version__", v, 0),
        rows_for(e, "__about__.__copyright__", VALUE, "__copyright__",
                 about, 0),
    ]
    df = pd.concat(parts, ignore_index=True)
    leaf = df["symbol"].str.rsplit(".", n=1).str[-1]
    text = df["explanation"].fillna("")
    bare_in_module = text.str.contains(
        r"(?:^|:\s)(?:version|VERSION|version_tuple): Attribute value",
        regex=True)
    value = df["kind"].eq(VALUE)
    unset = text.str.endswith("-> unset")
    rules = {
        "f1_old": leaf.isin(DUNDERS | BARE),
        "f1_new": value & ~unset & (leaf.isin(DUNDERS)
                                    | (leaf.isin(BARE) & bare_in_module)),
        "m_old": value & leaf.isin(METADATA_LEAVES),
        "m_new": value & ~unset & leaf.isin(METADATA_LEAVES),
    }
    return df, rules, keys


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


def emptied(df: pd.DataFrame, gone: pd.Series) -> int:
    pairs = df[GROUP].drop_duplicates()
    return len(pairs) - len(df.loc[~gone, GROUP].drop_duplicates())


def clear(df: pd.DataFrame, gone: pd.Series) -> set:
    every = set(df[GROUP].itertuples(index=False, name=None))
    return every - set(df.loc[~gone, GROUP].itertuples(index=False,
                                                       name=None))


def gates_here(df: pd.DataFrame, label: str) -> tuple[int, int, int]:
    p = df.groupby(GROUP)[label].agg(["size", "sum"])
    has = p["sum"] > 0
    return (int(p["sum"].sum()), int((has & (p["size"] > 10)).sum()),
            int((has & (p["size"] > 20)).sum()))


def case_build(tmp: pathlib.Path, df: pd.DataFrame, rules: dict,
               keys: dict) -> pd.DataFrame | None:
    print("\n2. build.py DROPS EXACTLY THOSE ROWS, BEFORE THE FEATURES")
    data = tmp / "data"
    data.mkdir()
    df.to_csv(data / "labelled.csv", index=False)
    code, out = run(tmp, "ml/features/build.py")
    check("build.py exits cleanly", code, 0)
    if code:
        print(out[-2000:])
        return None
    f1, meta = rules["f1_new"], rules["m_new"]
    n, m = int(f1.sum()), int(meta.sum())
    gone_f1 = emptied(df, f1)
    check(f"it reports dropping {n} version-string rows and the "
          f"{gone_f1} upgrades they emptied, and {m} metadata rows",
          (f"F1: {n:,} version-string rows dropped" in out,
           f"{gone_f1:,} upgrades held nothing else" in out,
           f"F39: {m:,} package-metadata rows dropped" in out),
          (True, True, True))
    k = {c: str for c in GROUP}
    built = pd.concat([pd.read_csv(data / "features.csv", dtype=k),
                       pd.read_csv(data / "holdout.csv", dtype=k)],
                      ignore_index=True)
    check("no row the rule names reaches features.csv or holdout.csv",
          int(version_strings(built).sum()), 0)
    check("every other row is there, once", len(built),
          len(df) - int((f1 | meta).sum()))
    leaf = built["symbol"].str.rsplit(".", n=1).str[-1]
    in_class = built["symbol"].str.endswith(".core.Config.VERSION")
    unset = built["explanation"].fillna("").str.endswith("-> unset")
    check("a removed __version__ (4), a class's VERSION (3), a removed "
          "submodule `version` (1), a VERSION with no explanation (1), and "
          "a __version__ and a __copyright__ changed to unset (1 each) stay",
          (int((leaf.eq("__version__") & built["kind"].eq("OBJECT_REMOVED"))
               .sum()),
           int(in_class.sum()),
           int((leaf.eq("version") & built["kind"].eq("OBJECT_REMOVED"))
               .sum()),
           int((leaf.eq("VERSION") & built["kind"].eq(VALUE)
                & ~in_class).sum()),
           int((leaf.eq("__version__") & unset).sum()),
           int((leaf.eq("__copyright__") & unset).sum())),
          (4, 3, 1, 1, 1, 1))
    left = set(built[GROUP].itertuples(index=False, name=None))
    check("the upgrades with only a version string or metadata are gone "
          "(A, B, E); the ones with a removal or a class's VERSION are not "
          "(C, D)",
          {x: keys[x] in left for x in "ABCDE"},
          {"A": False, "B": False, "C": True, "D": True, "E": False})
    size = built.groupby(GROUP)["symbol"].transform("size")
    check("release_size counts only the rows left, so the drop came first",
          bool((built["release_size"] == size).all()), True)
    return pd.read_csv(data / "holdout.csv", dtype=k)


def case_count(tmp: pathlib.Path, df: pd.DataFrame, rules: dict,
               keys: dict, built_holdout: pd.DataFrame | None) -> None:
    print("\n3. THE COUNT, ON THE SAME labelled.csv")
    data = tmp / "count" / "data"
    data.mkdir(parents=True)
    df.to_csv(data / "labelled.csv", index=False)
    before = digest(data)
    code, out = run(tmp / "count", "scripts/version_count.py")
    check("version_count.py runs", code, 0)
    if code:
        print(out[-2500:])
        return
    check("and writes nothing: every file in data/ is byte for byte the same",
          digest(data), before)

    held = holdout_mask(df)
    alias = df["label_alias"]
    old = rules["f1_old"] | rules["m_old"]
    new = rules["f1_new"] | rules["m_new"]
    for name, rule in (("F1 until 10 Oct", rules["f1_old"]),
                       ("F1 from 10 Oct", rules["f1_new"]),
                       ("F39 until 10 Oct", rules["m_old"]),
                       ("F39 from 10 Oct", rules["m_new"])):
        check(f"section 1, {name}: rows, dev, used in dev, holdout, "
              "upgrades emptied",
              number(name + r"\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)"
                     r"\s+([\d,]+)", out),
              (int(rule.sum()), int((rule & ~held).sum()),
               int(alias[rule & ~held].sum()), int((rule & held).sum()),
               emptied(df, rule)))
    for name, f1, both in (("until 10 Oct", rules["f1_old"], old),
                           ("from 10 Oct", rules["f1_new"], new)):
        e = emptied(df, both)
        check(f"section 1: F1 and F39 {name} together empty {e}, "
              f"{e - emptied(df, f1)} more than F1",
              number(rf"F1 and F39 {name} together empty ([\d,]+) upgrades, "
                     r"([\d,]+) more", out),
              (e, e - emptied(df, f1)))

    sec2 = out.split("2. WHAT COMES BACK")[-1].split("3. WHAT F1")[0]

    def line(name: str, why: str) -> tuple | None:
        return number(rf"{re.escape(name)}\s+{re.escape(why)}\n\s+dev "
                      r"([\d,]+) \(([\d,]+) used\), holdout ([\d,]+)", sec2)

    unset = "value changed to unset: a removal in effect"
    check("section 2: a removed __version__, 2 in dev (1 used), 1 in the "
          "holdout (its use not counted), and C's",
          line("__version__", "not a changed value: OBJECT_REMOVED"),
          (3, 1, 1))
    check("section 2: a class's VERSION, not directly in a module",
          line("VERSION", "not directly in a module"), (2, 0, 1))
    check("section 2: a VERSION griffe's explanation cannot place",
          line("VERSION", "griffe's explanation cannot be read"), (1, 0, 0))
    check("section 2: a removed submodule `version`",
          line("version", "not a changed value: OBJECT_REMOVED"), (1, 0, 0))
    check("section 2: a __version__ and a __copyright__ changed to unset",
          (line("__version__", unset), line("__copyright__", unset)),
          ((1, 1, 0), (1, 0, 0)))
    back = old & ~new
    check("section 2: in all",
          number(r"in all ([\d,]+) rows: dev ([\d,]+) \(([\d,]+) used\), "
                 r"holdout ([\d,]+)", sec2),
          (int(back.sum()), int((back & ~held).sum()),
           int(alias[back & ~held].sum()), int((back & held).sum())))

    sec3 = out.split("3. WHAT F1 STILL TAKES")[-1].split("4. THE ALL")[0]
    leaf = df["symbol"].str.rsplit(".", n=1).str[-1]
    symbols = set(df.loc[rules["f1_new"] & leaf.isin(BARE), "symbol"])
    check("section 3: _version.version in a module, 3 in dev and 2 in the "
          "holdout (B's among them)",
          number(r"version\s+dev\s+([\d,]+) \(0 used\)\s+holdout\s+([\d,]+)",
                 sec3), (3, 2))
    check(f"section 3 lists every symbol behind the bare names "
          f"({len(symbols)})",
          (f"({len(symbols)} symbols," in sec3,
           sorted(s for s in symbols if f"    {s}   " not in sec3)),
          (True, []))

    was, now = clear(df, old), clear(df, new)
    in_held = set(df.loc[held, GROUP].itertuples(index=False, name=None))
    check("section 4: the all-clear before and after, dev and holdout",
          number(r"until 10 Oct ([\d,]+), from 10 Oct ([\d,]+) \(([\d,]+) in "
                 r"dev, ([\d,]+) in the\s+holdout\)", out),
          (len(was), len(now), len(now - in_held), len(now & in_held)))
    check("which is A, B and E, and C and D leave it",
          (now == {keys["A"], keys["B"], keys["E"]},
           was - now == {keys["C"], keys["D"]}), (True, True))
    sec4 = out.split("4. THE ALL-CLEAR")[-1].split("5. WHAT THE")[0]
    check("section 4 names the two that leave, and why",
          (f"{keys['C'][0]} 9.9.9 -> 9.9.10: __version__ OBJECT_REMOVED"
           in sec4,
           f"{keys['D'][0]} 9.9.9 -> 9.9.10: VERSION {VALUE}" in sec4),
          (True, True))
    first = re.search(r"#(\d+)\s+(\S+) (\S+) -> (\S+)\s+released", sec4)
    check("the first preset is the most downloaded package's all-clear "
          "release: A, only __version__",
          (first and (first.group(1), first.group(2), first.group(4)),
           bool(re.search(r"#1\s+\S+ 9\.9\.9 -> 9\.9\.10 .* 1 row: "
                          r"__version__", sec4))),
          (("1", keys["A"][0], "9.9.10"), True))

    check("section 5: what the loader stops writing, before and after",
          number(r"in this file: ([\d,]+) \(the old rules' ([\d,]+), "
                 r"([\d,]+) fewer\)", out),
          (int(new.sum()), int(old.sum()), int(old.sum()) - int(new.sum())))

    h = df[held]
    hb, ha = h[~old[held]], h[~new[held]]
    check("section 6: rows and emptied upgrades that come back into the "
          "holdout",
          number(r"([\d,]+) rows? come back into it, and ([\d,]+) upgrades?",
                 out), (int((back & held).sum()), 1))
    m = re.search(r"pairs ([\d,]+) -> ([\d,]+), fingerprint (\w+) -> (\w+)",
                  out)
    check("section 6: pairs and fingerprint, before and after",
          m and (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)),
          (len(hb[GROUP].drop_duplicates()), len(ha[GROUP].drop_duplicates()),
           fingerprint(hb), fingerprint(ha)))
    if built_holdout is not None:
        check("and the fingerprint after is the one build.py's holdout.csv "
              "has", m and m.group(4), fingerprint(built_holdout))
    for lab in ("label_alias", "label"):
        check(f"section 6: the holdout's gates under {lab}, before and after",
              number(lab + r"\s+now (\d+)/(\d+)/(\d+), after (\d+)/(\d+)/"
                     r"(\d+)", out),
              gates_here(hb, lab) + gates_here(ha, lab))


def case_refusals(tmp: pathlib.Path, df: pd.DataFrame) -> None:
    print("\n4. WHAT THE COUNT REFUSES")
    empty = tmp / "empty"
    (empty / "data").mkdir(parents=True)
    code, out = run(empty, "scripts/version_count.py")
    check("no labelled.csv", (code != 0, "not found" in out), (True, True))
    df.drop(columns="explanation").to_csv(empty / "data" / "labelled.csv",
                                          index=False)
    code, out = run(empty, "scripts/version_count.py")
    check("a labelled.csv without griffe's explanation",
          (code != 0, "no explanation column" in out), (True, True))

    vc = load("version_count")
    good = tmp / "count"
    here = pathlib.Path.cwd()

    def main_says() -> str:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                vc.main()
        except SystemExit as e:
            return str(e.code)
        return ""

    try:
        os.chdir(good)
        vc.VERSION_BARE = {"version", "VERSION"}
        check("a build.py whose names changed",
              "not the six" in main_says(), True)
        vc.VERSION_BARE = BARE
        real = vc.version_strings
        vc.version_strings = lambda d: pd.Series(True, index=d.index)
        check("a rule that takes a row the old one did not",
              "can only take fewer" in main_says(), True)
        vc.version_strings = real
    finally:
        os.chdir(here)


def main() -> None:
    case_rule()
    df, rules, keys = fixture()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-version-"))
    try:
        built_holdout = case_build(tmp, df, rules, keys)
        case_count(tmp, df, rules, keys, built_holdout)
        case_refusals(tmp, df)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. F1 takes a version constant's changed value,")
    print("the bare names only in a module, and the count says what it took.")


if __name__ == "__main__":
    main()
