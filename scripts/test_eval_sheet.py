"""
Does scripts/eval_sheet.py put beside each drawn call what §38.18 says
the reader needs: the call before and after, the rule's printed reason,
the old and the new signature, and the source around it?

    python scripts/test_eval_sheet.py

Expected, written BEFORE the code ran, on the fixture: the fixer run on
scripts/fixtures/rewrite_app against the fixture site and bundle (as in
scripts/test_eval_run.py), the "new" version being the same site (the
fixture has one), and a draw of two calls written by hand:
  cases.py:9:5   a = Client("localhost", 6380, 2)      rewritten
     after   Client("localhost", port=6380, db=2, retries=3)
     symbol  alphalib.core.Client.__init__
     old     __init__(self, host, port=6379, db=0, retries=3)
     rules   R1, pass by name: ...; R2, pin the old default: retries=3 ...
  cases.py:21:5  n = connect("h", 2, timeout=9)        withheld
     symbol  alphalib.core.connect
     old     connect(host, port=1, timeout=5)
The "new" version is a copy of the site with retries=5 and timeout=7, so
that the two signatures visibly come from two directories.
Signatures are rendered as Python writes them: `/` after the
positional-only parameters, `*` before the keyword-only ones, `*args`
and `**kw` as such; a function the environment cannot answer renders
as "not read". The sheet text numbers the calls from 1 and holds the
three source lines around each; the csv has one row per call with
empty suggested, reason, verdict and note columns. A sample without a
tree in the manifest gets no context; two symbols, the first that
answers. 9 checks.
"""

import csv
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scripts.eval_run as er  # noqa: E402
import scripts.eval_sheet as sh  # noqa: E402
from fixer.oldapi import Param  # noqa: E402
from scripts.test_fixer_locate import (  # noqa: E402
    changes, extra_changes, r2_changes, rewrite_changes, write_bundle)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE / "fixtures" / "rewrite_app"
SITE = HERE / "fixtures" / "site"

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name, got, want):
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


def main() -> None:
    print("\n1. A SIGNATURE AS PYTHON WRITES IT")
    check("positional-only, keyword-only, *args and **kw rendered; defaults "
          "kept as written",
          (sh.render("f", [Param("a", "positional_only", None),
                           Param("b", "positional_only", "1"),
                           Param("c", "positional_or_keyword", "2"),
                           Param("d", "keyword_only", "1e3")]),
           sh.render("g", [Param("x", "positional_or_keyword", None),
                           Param("args", "var_positional", None),
                           Param("k", "keyword_only", "None"),
                           Param("kw", "var_keyword", None)]),
           sh.render("h", None)),
          ("f(a, b=1, /, c=2, *, d=1e3)", "g(x, *args, k=None, **kw)",
           "not read"))

    print("\n2. THE SHEET ON THE FIXTURE")
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        bundle = tmp / "b.sqlite"
        write_bundle(bundle, changes() + extra_changes() + rewrite_changes()
                     + r2_changes())
        root = tmp / "eval"
        tree = tmp / "tree"
        shutil.copytree(PROJECT, tree)
        run = ("rewrite_app", "alphalib", "1.0.0", "2.0.0")
        er.run_one([sys.executable, "-m", "fixer"], run, tree, SITE, root,
                   timeout=120, extra=["--bundle", str(bundle)])
        manifest = er.read_manifest(root)
        manifest["trees"]["rewrite_app"] = {"tree": str(tree)}
        er.write_manifest(root, manifest)
        with (root / "draw.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "sample", "changed", "from", "to", "file",
                        "line", "column", "use", "symbol", "call"])
            w.writerow(["rewritten", "rewrite_app", "alphalib", "1.0.0",
                        "2.0.0", "cases.py", 9, 5, "call",
                        "alphalib.core.Client.__init__",
                        'Client("localhost", port=6380, db=2, retries=3)'])
            w.writerow(["withheld", "rewrite_app", "alphalib", "1.0.0",
                        "2.0.0", "cases.py", 21, 5, "call",
                        "alphalib.core.connect", ""])
        new_site = tmp / "site2"
        shutil.copytree(SITE, new_site)
        core = new_site / "alphalib" / "core.py"
        core.write_text(core.read_text().replace("retries=3", "retries=5")
                        .replace("timeout=5", "timeout=7"))
        envs = {("alphalib", "1.0.0"): SITE, ("alphalib", "2.0.0"): new_site}
        rows = sh.build(root, envs)
        check("one row per drawn call, numbered, with the call before and "
              "after, the symbol and the kind",
              [(r["id"], r["kind"], r["file"], r["line"], r["before"],
                r["after"], r["symbol"]) for r in rows],
              [(1, "rewritten", "cases.py", 9,
                'a = Client("localhost", 6380, 2)',
                'Client("localhost", port=6380, db=2, retries=3)',
                "alphalib.core.Client.__init__"),
               (2, "withheld", "cases.py", 21,
                'n = connect("h", 2, timeout=9)',
                "", "alphalib.core.connect")])
        check("the old and the new signature beside each call, each from "
              "its own directory",
              [(r["old_signature"], r["new_signature"]) for r in rows],
              [("__init__(self, host, port=6379, db=0, retries=3)",
                "__init__(self, host, port=6379, db=0, retries=5)"),
               ("connect(host, port=1, timeout=5)",
                "connect(host, port=1, timeout=7)")])
        check("the rule lines as printed, joined by a separator the report "
              "never prints (a withheld line holds ' | would be:'), and the "
              "source around the call",
              (rows[0]["rules"].startswith("R1, pass by name:"),
               "R2, pin the old default: retries=3" in rows[0]["rules"],
               rows[1]["rules"].startswith("R1 withheld:"),
               len(rows[1]["rules"].split(sh.SEP)),
               "| would be:" in rows[1]["rules"],
               rows[0]["context"].splitlines()[1],
               len(rows[0]["context"].splitlines())),
              (True, True, True, 1, True,
               '     9 | a = Client("localhost", 6380, 2)', 3))
        no_tree = er.read_manifest(root)
        del no_tree["trees"]["rewrite_app"]
        er.write_manifest(root, no_tree)
        here = os.getcwd()
        os.chdir(tree)
        try:
            bare = sh.build(root, envs)
        finally:
            os.chdir(here)
        er.write_manifest(root, manifest)
        check("a sample whose tree the manifest does not name gets no source "
              "context, not the working directory's (here, the tree itself)",
              [r["context"] for r in bare], ["", ""])
        api = sh.api_for("alphalib", SITE)
        fake = root / "fake"
        fake.mkdir()
        entry = ("       cases.py:21:5                resolved    call      "
                 "n = connect(\"h\", 2, timeout=9)\n" + " " * 36
                 + "-> withheld\n" + " " * 39 + "R1 withheld: a reason\n")
        (fake / "alphalib-1.0.0-2.0.0.txt").write_text(
            "alphalib 1.0.0 -> 2.0.0: 1 release(s) crossed, 45 change(s); "
            "bundle built x\n\nproject /p: 1 file(s) read, 1 mention the "
            "import root(s) alphalib\n  1 line(s) use 1 of the 45 change(s), "
            "1 hit(s): 1 resolved, 0 referenced, 0 unverified.\n\nPART 1\n\n"
            "    1  alphalib.nowhere.f   used by 1   (1 line(s))\n" + entry
            + "\n    2  alphalib.core.connect   used by 1   (1 line(s))\n"
            + entry + "\nR1, pass by name: 0 call(s) rewritten, 1 withheld, "
            "0 need nothing. R2, pin the old default: 0 call(s) pinned, 0 "
            "withheld, 0 left to you, 0 need nothing.\n")
        with_fake = er.read_manifest(root)
        with_fake["trees"]["fake"] = {"tree": str(tree)}
        er.write_manifest(root, with_fake)
        with (root / "draw.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "sample", "changed", "from", "to", "file",
                        "line", "column", "use", "symbol", "call"])
            w.writerow(["withheld", "fake", "alphalib", "1.0.0", "2.0.0",
                        "cases.py", 21, 5, "call", "alphalib.nowhere.f", ""])
        two = sh.build(root, envs)
        check("a call under two symbols takes the first signature a version "
              "can answer, on the sheet and in first_read; none answering is "
              "'not read'",
              (two[0]["symbol"], two[0]["old_signature"],
               sh.first_read(api, ["alphalib.nowhere.f"])),
              ("alphalib.nowhere.f / alphalib.core.connect",
               "connect(host, port=1, timeout=5)", "not read"))
        text, out_csv = sh.write(root, rows)
        with out_csv.open(encoding="utf-8", newline="") as f:
            back = list(csv.DictReader(f))
        check("sheets.txt numbers the calls and names what to fill in; "
              "sheet.csv has the empty columns to fill",
              (text.is_file(), "#1  " in text.read_text(),
               "#2  " in text.read_text(),
               [(r["id"], r["suggested"], r["reason"], r["verdict"],
                 r["note"]) for r in back]),
              (True, True, True,
               [("1", "", "", "", ""), ("2", "", "", "", "")]))
        check("a call the report does not hold is named",
              sh.missing(root, [("rewrite_app", "alphalib", "1.0.0", "2.0.0",
                                 "cases.py", 999, 1, "call", "", "x")]),
              ["rewrite_app/alphalib/1.0.0/2.0.0 cases.py:999:1"])
        with (root / "draw.csv").open("a", newline="") as f:
            csv.writer(f).writerow(["rewritten", "rewrite_app", "alphalib",
                                    "1.0.0", "2.0.0", "cases.py", 999, 1,
                                    "call", "x", ""])
        try:
            sh.build(root, envs)
            refused = None
        except SystemExit as e:
            refused = str(e)
        check("and the sheet is refused while the draw holds one",
              refused is not None and "cases.py:999:1" in refused, True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The sheet carries what the reader needs.")


if __name__ == "__main__":
    main()
