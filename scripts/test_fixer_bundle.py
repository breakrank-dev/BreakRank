"""
Does the fixer's bundle hold exactly what the loader writes, and does the
reader answer only what the bundle knows? (NOTES §38.7, §38.2)

    python scripts/test_fixer_bundle.py

No database, no network: four small CSV files are written to a temporary
directory, the builder runs on them through the loader's own functions,
and the reader is checked against the numbers written here BEFORE the
code ran:

  the fixture   alpha: 1.0.0 -> 1.1.0 six rows (a moved parameter, a
                changed default, the version string, a removed helper, a
                changed constant, and the helper again under the same key),
                1.1.0 -> 1.2.0 one removed parameter with two export
                paths; releases.csv gives alpha 1.0.0 no_baseline, 1.1.0
                analysed 6 (the ingest's raw count, so the files agree),
                1.2.0 analysed 1, 1.3.0 analysed_clean, 1.3.5 yanked, and
                0.9.0 analysed_clean before them all.
                beta: 2.0 -> 2.1 and 2.1 -> 2.10 one removal each;
                releases.csv 2.0 no_baseline, 2.1 analysed 1, 2.9
                analysed_clean, 2.10 analysed 1 (2.10 is newer than 2.9).
                gamma: 0.1 -> 0.2 one removal, no releases.csv row.
                delta: releases.csv only, 3.0 no_baseline, 3.1 analysed_clean.
                packages.csv lists alpha, beta, delta; usage.csv counts
                alpha.core.Client.__init__ 12, alpha.util.helper 3, beta.b 1.
  expected      packages 4 (gamma from changes.csv); releases 14: analysed
                5, no_baseline 3, analysed_clean 4, yanked 1, unknown 1
                (gamma 0.1, the older side of its only pair); changes 8,
                1 row not a change (the version string), 1 duplicate key
                dropped; alpha 1.1.0 holds 4 changes with n_changes 4.
  the order     within alpha 1.1.0, public depth 1 for the constant and 2
                for the rest, so z_limit first though its name sorts last,
                then the two Client.__init__ rows (default before moved, by
                kind), then helper.
  the values    the default row (3, None); the constant (10, 20); the
                removal none. user_count 12, 12, 3, 0.
  the range     alpha 1.0.0 -> 1.2.0 crosses 1.1.0 and 1.2.0; beta 2.1 ->
                2.10 crosses 2.9 and 2.10 (not text order); 1.2.0 -> 1.2.0
                is refused as nothing to upgrade; a version or a package
                the bundle lacks is refused with the newest it holds.
"""

import contextlib
import io
import pathlib
import sqlite3
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fixer.bundle as fb  # noqa: E402
import fixer.cli as cli  # noqa: E402
import ml.db as db  # noqa: E402
import scripts.build_bundle as bb  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


# ------------------------------------------------------------------ fixture

COLS = ["package", "package_rank", "version_from", "version_to", "symbol",
        "kind", "sub_target", "is_private", "in_dunder_all", "module_depth",
        "is_top_level", "name_length", "released_at", "explanation",
        "export_paths", "inherited_by", "was_deprecated_before",
        "prior_breaks_in_module"]


def row(pkg, rank, vf, vt, sym, kind, msg, sub="", exports="", private=False):
    return {"package": pkg, "package_rank": rank, "version_from": vf,
            "version_to": vt, "symbol": sym, "kind": kind, "sub_target": sub,
            "is_private": private, "in_dunder_all": False,
            "module_depth": sym.count("."),
            "is_top_level": sym.count(".") == 1,
            "name_length": len(sym.rsplit(".", 1)[-1]),
            "released_at": "2026-03-01",
            "explanation": f"data/sdists/{pkg}/{vt}/x.py:3: {msg}",
            "export_paths": exports, "inherited_by": 0,
            "was_deprecated_before": False, "prior_breaks_in_module": 0}


def changes() -> pd.DataFrame:
    init = "alpha.core.Client.__init__"
    return pd.DataFrame([
        row("alpha", 1, "1.0.0", "1.1.0", init, "PARAMETER_MOVED",
            "Client.__init__(timeout): Positional parameter was moved",
            sub="timeout", exports="alpha.Client.__init__"),
        row("alpha", 1, "1.0.0", "1.1.0", init, "PARAMETER_CHANGED_DEFAULT",
            "Client.__init__(retries): Parameter default was changed: "
            "3 -> None",
            sub="retries", exports="alpha.Client.__init__"),
        row("alpha", 1, "1.0.0", "1.1.0", "alpha.__version__",
            "ATTRIBUTE_CHANGED_VALUE",
            "__version__: Attribute value was changed: '1.0.0' -> '1.1.0'"),
        row("alpha", 1, "1.0.0", "1.1.0", "alpha.util.helper",
            "OBJECT_REMOVED", "helper: Public object was removed"),
        row("alpha", 1, "1.0.0", "1.1.0", "alpha.z_limit",
            "ATTRIBUTE_CHANGED_VALUE",
            "z_limit: Attribute value was changed: 10 -> 20"),
        row("alpha", 1, "1.0.0", "1.1.0", "alpha.util.helper",
            "OBJECT_REMOVED",
            "helper: Public object was removed (through another path)"),
        row("alpha", 1, "1.1.0", "1.2.0", "alpha.core.Client.send",
            "PARAMETER_REMOVED", "Client.send(verify): Parameter was removed",
            sub="verify",
            exports="alpha.Client.send;alpha.client.Client.send"),
        row("beta", 2, "2.0", "2.1", "beta.b", "OBJECT_REMOVED",
            "b: Public object was removed"),
        row("beta", 2, "2.1", "2.10", "beta.c", "OBJECT_REMOVED",
            "c: Public object was removed"),
        row("gamma", 7, "0.1", "0.2", "gamma.g", "OBJECT_REMOVED",
            "g: Public object was removed"),
    ], columns=COLS)


RELEASES_CSV = """package,version,released_at,status,n_changes
alpha,0.9.0,2026-01-01,analysed_clean,0
alpha,1.0.0,2026-02-01,no_baseline,
alpha,1.1.0,2026-03-01,analysed,6
alpha,1.2.0,2026-04-01,analysed,1
alpha,1.3.0,2026-05-01,analysed_clean,0
alpha,1.3.5,2026-05-02,yanked,
beta,2.0,2026-01-01,no_baseline,
beta,2.1,2026-02-01,analysed,1
beta,2.9,2026-03-01,analysed_clean,0
beta,2.10,2026-04-01,analysed,1
delta,3.0,2026-01-01,no_baseline,
delta,3.1,2026-02-01,analysed_clean,0
"""

USAGE_CSV = """symbol,user_count
alpha.core.Client.__init__,12
alpha.util.helper,3
beta.b,1
"""

PACKAGES_CSV = """package,download_rank,github_repo
alpha,1,o/alpha
beta,2,
delta,4,
"""


@contextlib.contextmanager
def fixture_files():
    """The four CSVs in a temporary directory, with ml.db pointed at them
    for the duration (the test's own files, never data/)."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-bundle-"))
    changes().to_csv(tmp / "changes.csv", index=False)
    (tmp / "releases.csv").write_text(RELEASES_CSV)
    (tmp / "usage.csv").write_text(USAGE_CSV)
    (tmp / "packages.csv").write_text(PACKAGES_CSV)
    saved = (db.CHANGES, db.RELEASES, db.USAGE, db.PACKAGES,
             bb.CHANGES, bb.RELEASES, bb.USAGE, bb.PACKAGES)
    db.CHANGES = bb.CHANGES = tmp / "changes.csv"
    db.RELEASES = bb.RELEASES = tmp / "releases.csv"
    db.USAGE = bb.USAGE = tmp / "usage.csv"
    db.PACKAGES = bb.PACKAGES = tmp / "packages.csv"
    try:
        yield tmp
    finally:
        (db.CHANGES, db.RELEASES, db.USAGE, db.PACKAGES,
         bb.CHANGES, bb.RELEASES, bb.USAGE, bb.PACKAGES) = saved


def build(tmp: pathlib.Path) -> tuple[pathlib.Path, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sys.argv = ["build_bundle.py", "--out", str(tmp / "b.sqlite")]
        bb.main()
    return tmp / "b.sqlite", out.getvalue()


# -------------------------------------------------------------------- cases

def case_split() -> None:
    print("\n1. OLD AND NEW VALUES OUT OF GRIFFE'S WORDS")
    check("a changed default", fb.split_change(
        "f(x): Parameter default was changed: 3 -> None"), ("3", "None"))
    check("a changed attribute value", fb.split_change(
        "LIMIT: Attribute value was changed: 10 -> 20"), ("10", "20"))
    check("a value that contains the arrow itself splits where both sides "
          "parse", fb.split_change(
              "f(sep): Parameter default was changed: ', ' -> ' -> '"),
          ("', '", "' -> '"))
    check("a name, not a literal, is still a value (R2 decides later what "
          "is plain)", fb.split_change(
              "f(x): Parameter default was changed: lib.no_default -> None"),
          ("lib.no_default", "None"))
    check("a lambda default parses as an expression too", fb.split_change(
        "f(x): Parameter default was changed: (lambda: (a,))() -> None"),
        ("(lambda: (a,))()", "None"))
    check("no arrow, nothing claimed",
          fb.split_change("f(x): Parameter was removed"), None)
    check("no valid split, nothing claimed", fb.split_change(
        "f(x): Parameter default was changed: 1 -> 2 -> 3"), None)
    check("two valid splits (comments make both sides parse), nothing "
          "claimed", fb.split_change(
              "f(x): Parameter default was changed: 1 # -> 2 -> 3 # -> 4"),
          None)


def case_versions() -> None:
    print("\n2. VERSION ORDER IS PACKAGING'S, TEXT ORDER AFTER")
    check("2.10 sorts after 2.9, a pre-release before, text last",
          sorted(["2.10", "2.9", "abc", "2.1", "1.0rc1"], key=fb.version_key),
          ["1.0rc1", "2.1", "2.9", "2.10", "abc"])


def case_build(tmp: pathlib.Path) -> fb.Bundle:
    print("\n3. THE BUILDER WRITES WHAT THE LOADER WRITES")
    path, out = build(tmp)
    b = fb.Bundle(path)
    check("counts: 4 packages, 14 releases, 8 changes",
          b.counts(), {"packages": 4, "releases": 14, "changes": 8})
    check("gamma, in changes.csv only, is a package; delta, in releases.csv "
          "and packages.csv only, too", b.packages(),
          ["alpha", "beta", "delta", "gamma"])
    statuses = {}
    for p in b.packages():
        for r in b.releases(p):
            s = r.analysis_status
            statuses[s] = statuses.get(s, 0) + 1
    check("statuses as the loader establishes them, gamma 0.1 unknown",
          statuses, {"analysed": 5, "no_baseline": 3, "analysed_clean": 4,
                     "yanked": 1, "unknown": 1})
    check("compared means a verdict on the pair: analysed and analysed_clean "
          "only",
          sorted({r.analysis_status for p in b.packages()
                  for r in b.releases(p) if r.compared}),
          ["analysed", "analysed_clean"])
    alpha = {r.version: r for r in b.releases("alpha")}
    check("alpha 1.1.0 counts the 4 rows written, not the ingest's 6",
          (alpha["1.1.0"].analysis_status, alpha["1.1.0"].n_changes),
          ("analysed", 4))
    check("the builder says what it left out and dropped",
          ("1 rows not changes" in out, "1 duplicate key(s) dropped" in out,
           "packages        4" in out, "releases       14" in out,
           "changes         8" in out), (True, True, True, True, True))
    m = b.manifest()
    check("the manifest carries the schema, the counts, the sources' hashes "
          "and when",
          (m["schema_version"], m["n_changes"], m["n_releases"],
           m["n_packages"], len(m["sha256_changes"]), "built_at" in m,
           m["ranker"]),
          ("1", "8", "14", "4", 64, True, "path-length"))
    return b


def case_changes(b: fb.Bundle) -> None:
    print("\n4. THE CHANGES, IN THE SITE'S ORDER, WITH THEIR VALUES")
    got = b.changes("alpha", ["1.1.0"])
    check("alpha 1.1.0: the constant first (depth 1, though last by name), "
          "then the two Client.__init__ rows by kind, then helper; the "
          "version string and the duplicate are not there",
          [(c.symbol, c.kind, c.score) for c in got],
          [("alpha.z_limit", "ATTRIBUTE_CHANGED_VALUE", -1.0),
           ("alpha.core.Client.__init__", "PARAMETER_CHANGED_DEFAULT", -2.0),
           ("alpha.core.Client.__init__", "PARAMETER_MOVED", -2.0),
           ("alpha.util.helper", "OBJECT_REMOVED", -2.0)])
    check("public depth reads the export paths: Client.__init__ is defined "
          "at depth 3 and reached at 2",
          [(c.module_depth, c.public_depth) for c in got],
          [(1, 1), (3, 2), (3, 2), (2, 2)])
    check("old and new values where griffe's words carry them",
          [(c.old_value, c.new_value) for c in got],
          [("10", "20"), ("3", "None"), (None, None), (None, None)])
    check("user_count by the exact symbol, as the site joins it",
          [c.user_count for c in got], [0, 12, 12, 3])
    check("the message is griffe's words with the file location stripped",
          got[1].message,
          "Client.__init__(retries): Parameter default was changed: 3 -> None")
    check("sub_target rides along", [c.sub_target for c in got],
          ["", "retries", "timeout", ""])
    send = b.changes("alpha", ["1.2.0"])[0]
    check("every path downstream code can write: the definition path first, "
          "then the export paths",
          send.paths, ("alpha.core.Client.send", "alpha.Client.send",
                       "alpha.client.Client.send"))
    both = b.changes("alpha", ["1.1.0", "1.2.0"])
    check("two releases together keep the site's order across them",
          [c.symbol for c in both],
          ["alpha.z_limit", "alpha.core.Client.__init__",
           "alpha.core.Client.__init__", "alpha.core.Client.send",
           "alpha.util.helper"])
    check("no versions, no changes; all versions, all of the package's",
          (b.changes("alpha", []), len(b.changes("alpha"))), ([], 5))


def case_range(b: fb.Bundle) -> None:
    print("\n5. AN UPGRADE IS (from, to], IN VERSION ORDER, OR REFUSED")
    check("alpha 1.0.0 -> 1.2.0 crosses 1.1.0 and 1.2.0",
          [r.version for r in b.upgrade("alpha", "1.0.0", "1.2.0")],
          ["1.1.0", "1.2.0"])
    check("beta 2.1 -> 2.10 crosses 2.9 then 2.10, not text order",
          [r.version for r in b.upgrade("beta", "2.1", "2.10")],
          ["2.9", "2.10"])
    check("a yanked release in the range is crossed and says so",
          [(r.version, r.describe()) for r in
           b.upgrade("alpha", "1.3.0", "1.3.5")],
          [("1.3.5", "not compared: yanked on PyPI")])
    check("a range across a not-compared release crosses it",
          [(r.version, r.compared) for r in
           b.upgrade("alpha", "0.9.0", "1.1.0")],
          [("1.0.0", False), ("1.1.0", True)])
    check("the releases say what the bundle knows of each, in the site's "
          "words",
          [r.describe() for r in b.releases("beta")],
          ["not compared: the analysis window starts at this release",
           "compared with the one before it: 1 change(s)",
           "compared with the one before it: no breaking change",
           "compared with the one before it: 1 change(s)"])
    check("gamma 0.1 reads as unconfirmed, the site's wording for unknown",
          b.releases("gamma")[0].describe(),
          "we can't confirm it was compared with the one before it")

    def refused(fn):
        try:
            fn()
        except fb.BundleError as e:
            return str(e)
        return None

    check("the same version twice: nothing to upgrade, already installed",
          refused(lambda: b.upgrade("alpha", "1.2.0", "1.2.0")),
          "nothing to upgrade: alpha 1.2.0 is not older than 1.2.0. The new "
          "version is already installed.")
    check("a downgrade is refused",
          "is not older than" in (refused(
              lambda: b.upgrade("alpha", "1.2.0", "1.0.0")) or ""), True)
    msg = refused(lambda: b.upgrade("alpha", "1.0.0", "9.9")) or ""
    check("a version the bundle lacks: refused, with the newest it holds and "
          "when it was built, and the way forward",
          ("alpha 9.9 is not a release the bundle holds (--to)" in msg,
           "its newest alpha is 1.3.5" in msg, "built 20" in msg,
           "needs the ingest" in msg), (True, True, True, True))
    check("a package the bundle lacks is refused, not guessed",
          "has no package named 'zeta'" in (refused(lambda: b.releases("zeta"))
                                            or ""), True)
    check("newest is the last in version order", b.newest("beta").version,
          "2.10")


def case_guards(tmp: pathlib.Path) -> None:
    print("\n6. THE READER AND THE BUILDER REFUSE WHAT THEY CANNOT ANSWER")
    try:
        fb.Bundle(tmp / "nowhere.sqlite")
        got = None
    except fb.BundleError as e:
        got = str(e)
    check("a missing bundle names the builder and the environment variable",
          ("no bundle at" in (got or ""), "build_bundle.py" in (got or ""),
           fb.ENV_VAR in (got or "")), (True, True, True))
    old = tmp / "old.sqlite"
    conn = sqlite3.connect(old)
    conn.executescript(fb.SCHEMA)
    conn.execute("INSERT INTO manifest VALUES ('schema_version', '0')")
    conn.commit()
    conn.close()
    try:
        fb.Bundle(old)
        got = None
    except fb.BundleError as e:
        got = str(e)
    check("another schema version is refused, not misread",
          "bundle schema '0'" in (got or ""), True)
    # the oracle: a builder whose rows differ from the loader's stops
    saved = bb.change_records
    bb.change_records = lambda kept, rels, usage: saved(kept, rels, usage)[:-1]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sys.argv = ["build_bundle.py", "--out", str(tmp / "never.sqlite")]
            try:
                bb.main()
                code = None
            except SystemExit as e:
                code = str(e.code)
    finally:
        bb.change_records = saved
    check("a builder that would write other rows than the loader is refused "
          "and writes nothing",
          (code is not None and "REFUSING" in code and "7 changes" in code
           and "8" in code, (tmp / "never.sqlite").exists()), (True, False))


def case_cli(tmp: pathlib.Path) -> None:
    print("\n7. THE COMMAND LINE: RANK, AND SAY WHAT IS NOT HERE YET")
    bundle = str(tmp / "b.sqlite")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["alpha", "--from", "1.0.0", "--to", "1.2.0",
                         "--bundle", bundle])
    text = out.getvalue()
    check("exit 0 and the header with both releases crossed",
          (code, "alpha 1.0.0 -> 1.2.0: 2 release(s) crossed, 5 change(s)"
           in text), (0, True))
    check("each release with what the bundle says of it",
          ("1.1.0          compared with the one before it: 4 change(s)"
           in text,
           "1.2.0          compared with the one before it: 1 change(s)"
           in text),
          (True, True))
    check("the changes ranked, the constant first with its values and the "
          "default row with its parameter",
          ("    1  alpha.z_limit   ATTRIBUTE_CHANGED_VALUE   10 -> 20   "
           "used by 0" in text,
           "    2  alpha.core.Client.__init__ (retries)   "
           "PARAMETER_CHANGED_DEFAULT   3 -> None   used by 12" in text),
          (True, True))
    check("it says locate and the rewrites are not here, and credits griffe",
          ("locate and the rewrites are not in this version" in text,
           "changes found with griffe." in text), (True, True))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["beta", "--to", "2.10", "--bundle", bundle])
    check("--from defaults to the one-step upgrade, the release before --to",
          (code, "beta 2.9 -> 2.10: 1 release(s) crossed, 1 change(s)"
           in out.getvalue()), (0, True))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["alpha", "--from", "0.9.0", "--to", "1.3.5",
                         "--bundle", bundle])
    check("a range with not-compared releases (no_baseline, yanked) says "
          "the list cannot be complete for them",
          (code, "** 2 release(s) in the range were not compared" in
           out.getvalue()), (0, True))
    err, quiet = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(quiet):
        code = cli.main(["zeta", "--to", "1.0", "--bundle", bundle])
    check("an unknown package: exit 2 and the refusal on stderr",
          (code, "has no package named 'zeta'" in err.getvalue()), (2, True))
    err, quiet = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(quiet):
        code = cli.main(["alpha", "--bundle", str(tmp / "nowhere.sqlite")])
    check("a missing bundle: exit 2", (code, "no bundle at" in err.getvalue()),
          (2, True))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(["--list", "--bundle", bundle])
    text = out.getvalue()
    check("--list names the packages",
          (code, "4 package(s) in the bundle" in text, "  gamma" in text),
          (0, True, True))


def main() -> None:
    case_split()
    case_versions()
    with fixture_files() as tmp:
        b = case_build(tmp)
        case_changes(b)
        case_range(b)
        b.close()
        case_guards(tmp)
        case_cli(tmp)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not build or ship a bundle until this passes.")
        sys.exit(1)
    print("All checks passed. The bundle holds what the loader writes, and")
    print("the reader answers only what the bundle knows.")


if __name__ == "__main__":
    main()
