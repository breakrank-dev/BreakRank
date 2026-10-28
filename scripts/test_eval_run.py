"""
Does scripts/eval_run.py do what NOTES §38.18 says, on the fixture and
on hand-written inputs, without the network?

    python scripts/test_eval_run.py

Expected, written BEFORE the code ran.

The plan: from a sample csv and a bundle, one run per (sample package,
changed package, upgrade); a reserve or skipped row plans nothing; a
function the bundle has no rewritable row on is refused as a csv that
does not match the bundle. The fixture bundle holds one upgrade, so a
sample row naming two alphalib functions plans one run, one old
version (alphalib 1.0.0) and one new (2.0.0).

Choosing a release from PyPI's record: the newest FINAL release by PEP
440 (no pre-, dev- or yanked release) that has files; its sdist when it
has one, else its pure-Python wheel, else no source (the reserve rule).
  A  1.0.0 sdist; 1.1.0rc1 sdist; 2.0.0.dev1 sdist      -> 1.0.0, sdist
  B  1.0.0 sdist; 1.1.0 py3-none-any wheel only        -> 1.1.0, wheel
  C  1.0.0 sdist; 1.1.0 sdist yanked                   -> 1.0.0, sdist
  D  1.0.0 sdist; 1.1.0 cp312 manylinux wheel only     -> 1.1.0, none
  E  1.0.0 sdist; 1.0.0.post1 sdist                    -> 1.0.0.post1, sdist
  F  1.0.0 sdist; 1.1.0 with no files                  -> 1.0.0, sdist
  G  1.0.0 sdist; 1.1.0 wheel and sdist                -> 1.1.0, sdist

  H  1.0.0 sdist; 1.1.0 py2.py3-none-any wheel only    -> 1.1.0, wheel
     (pure Python under either tag; §38.18 names py3-none-any and means
     pure Python)

The report, parsed: the fixture project (scripts/fixtures/rewrite_app)
against the fixture site and bundle gives exit 1 and
  2 files read, 2 mention the import root; the old version read;
  48 lines use 38 of the 45 changes; 50 hits: 48 resolved, 0 referenced,
  2 unverified; R1 16 rewritten, 9 withheld, 4 need nothing; R2 26
  pinned, 10 withheld, 2 left to you, 3 need nothing; decision lines:
  32 rewritten, 13 withheld, 1 left to you, 2 need nothing; the diff
  holds 2 files.
These are scripts/test_fixer_rewrite.py's numbers. The unit counted is
a CALL: the report prints a call's decision under every entry whose
rows the call meets (a call resolving to two defining paths appears
twice; scripts/test_fixer_rewrite.py asserts as much), so the decision
lines are merged on (file, line, column, use, kind, call) with the
symbols collected. A report without hits parses to zeros; a run that
exits 2 parses to no counts and keeps the message; a report cut off
before its footer (a timeout) parses to no counts and an error; a
"[stderr]" trailer changes nothing; the "were not read" environment
line gives old_read False; a hit in a file whose path holds a space is
read. The command the harness runs is exactly §38.18's, with no
--apply.

The draw (§38.18): the budget B over runs with r_i rewritten calls.
Every run with fewer than m gives all of them, every other run at least
m, m the largest of 3, 2, 1 that fits the budget; what is left of the
budget goes in proportion to (r_i - m), largest remainder, capped at
r_i. Runs A 10, B 2, C 0, D 5:
  budget 8   -> A 3, B 2, C 0, D 3          (the minimums use it up)
  budget 12  -> A 6, B 2, C 0, D 4          (4 over, 7:2, A 3 D 1)
  budget 17+ -> all
  five runs of 4 each, budget 10 -> m falls to 2, each gives 2
  forty runs of 5, budget 30 -> one each does not fit: allocate answers
     None and the draw takes the 30 from all the calls as one list
     ("pooled"), proportional in expectation, never alphabetical
The calls drawn come from Random(seed) over each run's calls in file,
line order, runs in sorted order; the same seed draws the same calls,
another seed another set. A placed sdist is read back by
fixer/env.py's find_installed as the version and roots it names; a
distribution that is a single module (src/typing_extensions.py) is
placed as that module. pip's "No matching distribution found" is the
one failure that sends obtain to the sdist; a record that is skipped
or FAILED is not done. Two withheld calls chained on one line are two
decisions when their rule lines differ; decision lines that do not add
up to the footer are an error; pip's offline message is not 'no wheel';
a single module at the sdist's root is placed. 28 checks.
"""

import contextlib
import csv
import io
import json
import pathlib
import random
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scripts.eval_run as er  # noqa: E402
from fixer.env import find_installed  # noqa: E402
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


def sdist(version, yanked=False):
    return {"packagetype": "sdist", "filename": f"p-{version}.tar.gz",
            "url": f"https://x/p-{version}.tar.gz", "size": 10,
            "digests": {"sha256": "0" * 64}, "upload_time_iso_8601":
            "2026-01-01T00:00:00Z", "yanked": yanked}


def wheel(version, tag="py3-none-any"):
    return {"packagetype": "bdist_wheel",
            "filename": f"p-{version}-{tag}.whl",
            "url": f"https://x/p-{version}-{tag}.whl", "size": 10,
            "digests": {"sha256": "1" * 64}, "upload_time_iso_8601":
            "2026-01-01T00:00:00Z", "yanked": False}


def main() -> None:
    print("\n1. THE PLAN")
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        bundle = tmp / "b.sqlite"
        write_bundle(bundle, changes() + extra_changes() + rewrite_changes()
                     + r2_changes())
        sample = tmp / "sample.csv"
        with sample.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "scanner", "functions", "packages",
                        "upgrades", "role", "note", "detail"])
            w.writerow([1, "appa", 2, 1, 1, "sample", "",
                        "alphalib:alphalib.core.Client.__init__;"
                        "alphalib:alphalib.opts.fetch"])
            w.writerow([2, "appb", 1, 1, 1, "same source as appa, skipped",
                        "", "alphalib:alphalib.opts.fetch"])
            w.writerow([3, "appc", 1, 1, 1, "reserve", "",
                        "alphalib:alphalib.opts.ship"])
        runs, olds, news = er.plan(sample, bundle)
        check("one run per (sample package, changed package, upgrade); the "
              "skipped and the reserve plan nothing",
              (runs, olds, news),
              ([("appa", "alphalib", "1.0.0", "2.0.0")],
               [("alphalib", "1.0.0")], [("alphalib", "2.0.0")]))
        with sample.open("a", newline="") as f:
            csv.writer(f).writerow([4, "appd", 1, 1, 1, "sample", "",
                                    "alphalib:alphalib.util.LIMIT"])
        try:
            er.plan(sample, bundle)
            refused = None
        except er.PlanError as e:
            refused = str(e)
        check("a function the bundle has no rewritable row on is refused",
              refused is not None and "appd" in refused
              and "alphalib.util.LIMIT" in refused, True)

        print("\n2. WHICH RELEASE IS THE LATEST")
        cases = {
            "A": {"1.0.0": [sdist("1.0.0")], "1.1.0rc1": [sdist("1.1.0rc1")],
                  "2.0.0.dev1": [sdist("2.0.0.dev1")]},
            "B": {"1.0.0": [sdist("1.0.0")], "1.1.0": [wheel("1.1.0")]},
            "C": {"1.0.0": [sdist("1.0.0")],
                  "1.1.0": [sdist("1.1.0", yanked=True)]},
            "D": {"1.0.0": [sdist("1.0.0")],
                  "1.1.0": [wheel("1.1.0",
                                  "cp312-cp312-manylinux_2_17_x86_64")]},
            "E": {"1.0.0": [sdist("1.0.0")],
                  "1.0.0.post1": [sdist("1.0.0.post1")]},
            "F": {"1.0.0": [sdist("1.0.0")], "1.1.0": []},
            "G": {"1.0.0": [sdist("1.0.0")],
                  "1.1.0": [wheel("1.1.0"), sdist("1.1.0")]},
            "H": {"1.0.0": [sdist("1.0.0")],
                  "1.1.0": [wheel("1.1.0", "py2.py3-none-any")]},
        }
        got = {}
        for k, rel in cases.items():
            version, file, kind = er.choose_release({"releases": rel})
            got[k] = (version, kind, file["filename"] if file else None)
        check("the newest final release; its sdist, else its pure wheel, "
              "else no source",
              got,
              {"A": ("1.0.0", "sdist", "p-1.0.0.tar.gz"),
               "B": ("1.1.0", "wheel", "p-1.1.0-py3-none-any.whl"),
               "C": ("1.0.0", "sdist", "p-1.0.0.tar.gz"),
               "D": ("1.1.0", None, None),
               "E": ("1.0.0.post1", "sdist", "p-1.0.0.post1.tar.gz"),
               "F": ("1.0.0", "sdist", "p-1.0.0.tar.gz"),
               "G": ("1.1.0", "sdist", "p-1.1.0.tar.gz"),
               "H": ("1.1.0", "wheel", "p-1.1.0-py2.py3-none-any.whl")})

        print("\n3. ONE RUN ON THE FIXTURE, AND THE REPORT PARSED")
        root = tmp / "eval"
        tree = tmp / "tree"
        shutil.copytree(PROJECT, tree)
        run = ("rewrite_app", "alphalib", "1.0.0", "2.0.0")
        tool = [sys.executable, "-m", "fixer"]
        rec = er.run_one(tool, run, tree, SITE, root, timeout=120,
                         extra=["--bundle", str(bundle)])
        diff = root / "rewrite_app" / "alphalib-1.0.0-2.0.0.diff"
        check("the run: exit 1 (hits), its stdout and diff written where "
              "§38.18 says, the time recorded, the command §38.18's with no "
              "--apply",
              (rec["exit"], (root / "rewrite_app" /
                             "alphalib-1.0.0-2.0.0.txt").is_file(),
               diff.is_file(), rec["seconds"] > 0, rec["timed_out"],
               rec["command"][len(tool):]),
              (1, True, True, True, False,
               ["alphalib", str(tree), "--from", "1.0.0", "--to", "2.0.0",
                "--env", str(SITE), "--diff", str(diff), "--bundle",
                str(bundle)]))
        counts = er.parse_report(
            (root / "rewrite_app" / "alphalib-1.0.0-2.0.0.txt").read_text())
        check("the report parsed: files, lines, hits by confidence, the two "
              "footers, the decision lines, the diff's files",
              counts,
              {"files_read": 2, "files_mention": 2, "old_read": True,
               "lines": 48, "changes_used": 38, "changes": 45, "hits": 50,
               "resolved": 48, "referenced": 0, "unverified": 2,
               "r1_rewritten": 16, "r1_withheld": 9, "r1_nothing": 4,
               "r2_pinned": 26, "r2_withheld": 10, "r2_left": 2,
               "r2_nothing": 3, "rewritten": 32, "withheld": 13, "left": 1,
               "nothing": 2, "diff_files": 2, "error": ""})
        quiet = er.parse_report(
            "redis 7.4.1 -> 8.0.0: 1 release(s) crossed, 134 change(s); "
            "bundle built x\n\nproject /p: 3 file(s) read, 0 mention the "
            "import root(s) redis\n  old version redis 7.4.1 read from /e "
            "(import roots: redis): its classes' bases and signatures are "
            "known.\n  0 line(s) use 0 of the 134 change(s), 0 hit(s): 0 "
            "resolved, 0 referenced, 0 unverified.\n\nR1, pass by name: 0 "
            "call(s) rewritten, 0 withheld, 0 need nothing. R2, pin the old "
            "default: 0 call(s) pinned, 0 withheld, 0 left to you, 0 need "
            "nothing.\ndiff written to /d (0 file(s)); nothing to rewrite, "
            "so it is empty\n")
        check("a report without hits parses to zeros",
              (quiet["files_read"], quiet["files_mention"], quiet["hits"],
               quiet["rewritten"], quiet["r2_pinned"], quiet["diff_files"],
               quiet["error"]),
              (3, 0, 0, 0, 0, 0, ""))
        failed = er.parse_report("alphalib 9.9.9 is not a release the bundle "
                                 "holds (--to); its newest is 2.0.0.\n")
        check("a run that stopped parses to no counts and keeps the message",
              (failed["hits"], failed["rewritten"],
               "not a release the bundle holds" in failed["error"]),
              (None, None, True))
        head = ("alphalib 1.0.0 -> 2.0.0: 1 release(s) crossed, 45 change(s); "
                "bundle built x\n\nproject /p: 2 file(s) read, 2 mention the "
                "import root(s) alphalib\n  alphalib is not installed in /e. "
                "So the old version's bases and signatures were not read; a "
                "method called on an instance is listed as unverified where "
                "the bundle does not name its path.\n  1 line(s) use 1 of the "
                "45 change(s), 1 hit(s): 1 resolved, 0 referenced, 0 "
                "unverified.\n\nPART 1  changes your code uses\n\n    1  "
                "alphalib.core.helper   used by 1   (1 line(s))\n         "
                "PARAMETER_MOVED (b)\n       my tests/t.py:7:3           "
                "  resolved    call      x = helper(1, 2)\n" + " " * 36
                + "-> rewritten: helper(1, b=2)\n" + " " * 39
                + "R1, pass by name: b -> b=2\n")
        foot = ("\nR1, pass by name: 1 call(s) rewritten, 0 withheld, 0 need "
                "nothing. R2, pin the old default: 0 call(s) pinned, 0 "
                "withheld, 0 left to you, 0 need nothing; the diff is in "
                "/d.\ndiff written to /d (1 file(s))\n")
        whole = er.parse_report(head + foot + "\n[stderr]\nsome warning\n")
        cut = er.parse_report(head + "\n[stderr]\ntimed out after 600 s\n")
        check("the 'were not read' line gives old_read False; a [stderr] "
              "trailer changes nothing; a report cut before its footer "
              "parses to no counts and an error",
              ((whole["old_read"], whole["rewritten"], whole["r1_rewritten"],
                whole["error"]),
               (cut["files_read"], cut["rewritten"], cut["r1_rewritten"],
                "timed out" in cut["error"])),
              ((False, 1, 1, ""), (2, None, None, True)))
        spaced = er.decisions(head + foot)
        check("a hit in a file whose path holds a space is read",
              [(d["file"], d["line"], d["column"], d["kind"], d["call"])
               for d in spaced],
              [("my tests/t.py", 7, 3, "rewritten", "helper(1, b=2)")])
        twice = (head.replace("my tests/t.py", "t.py")
                 + "\n    2  alphalib.helper   used by 1   (1 line(s))\n"
                   "         PARAMETER_MOVED (b)\n       t.py:7:3            "
                   " resolved    call      x = helper(1, 2)\n" + " " * 36
                 + "-> rewritten: helper(1, b=2)\n" + " " * 39
                 + "R1, pass by name: b -> b=2\n" + foot)
        merged = er.decisions(twice)
        check("one call printed under two entries is one decision, with "
              "both symbols, and is counted once",
              (len(merged), merged[0]["symbols"],
               er.parse_report(twice)["rewritten"]),
              (1, ["alphalib.core.helper", "alphalib.helper"], 1))
        chained = (head.replace("my tests/t.py", "t.py")
                   .replace("-> rewritten: helper(1, b=2)", "-> withheld")
                   .replace("R1, pass by name: b -> b=2",
                            "R2 withheld: the first reason")
                   + "\n    2  alphalib.other   used by 1   (1 line(s))\n"
                     "         PARAMETER_CHANGED_DEFAULT (c)   1 -> 2\n"
                     "       t.py:7:3             resolved    method    "
                     "x = helper(1, 2).other()\n" + " " * 36
                   + "-> withheld\n" + " " * 39
                   + "R2 withheld: the second reason\n"
                   + foot.replace("1 call(s) rewritten, 0 withheld",
                                  "0 call(s) rewritten, 0 withheld")
                   .replace("0 call(s) pinned, 0 withheld",
                            "0 call(s) pinned, 2 withheld"))
        two = er.decisions(chained)
        check("two withheld calls on one line at one column are two "
              "decisions when their rule lines differ",
              ([(d["use"], d["rules"]) for d in two],
               er.parse_report(chained)["withheld"],
               er.parse_report(chained)["error"]),
              ([("call", ["R2 withheld: the first reason"]),
                ("method", ["R2 withheld: the second reason"])], 2, ""))
        off = er.parse_report(chained.replace("0 call(s) pinned, 2 withheld",
                                              "0 call(s) pinned, 3 withheld"))
        check("decision lines that do not add up to the footer are an error",
              (off["withheld"], "do not add up" in off["error"]),
              (2, True))
        manifest = er.read_manifest(root)
        manifest["runs"][er.key(run)] = rec
        skipped = ("rewrite_app", "zeta", "1.0", "2.0")
        manifest["runs"][er.key(skipped)] = {"exit": None,
                                             "skipped": "no old version"}
        er.write_manifest(root, manifest)
        rows = er.count(root, [run, skipped])
        check("count: one csv row per run with the parsed numbers and the "
              "exit code from the manifest; a skipped run carries its reason",
              ([(r["sample"], r["changed"], r["from"], r["to"], r["exit"],
                 r["hits"], r["rewritten"], r["withheld"], r["error"])
                for r in rows],
               (root / "counts.csv").is_file()),
              ([("rewrite_app", "alphalib", "1.0.0", "2.0.0", 1, 50, 32, 13,
                 ""),
                ("rewrite_app", "zeta", "1.0", "2.0", None, None, None, None,
                 "no old version")],
               True))
        lines = er.totals(rows)
        check("the totals: by sample package and by changed package, the "
              "unverified beside the hits as a count over a count, the runs "
              "without a report counted apart",
              ([ln.split()[:2] for ln in lines
                if ln.startswith("rewrite_app")],
               [ln.split()[:2] for ln in lines if ln.startswith("alphalib")],
               any("2/50" in ln for ln in lines),
               any(ln.startswith("ALL") and ln.rstrip().endswith("1")
                   for ln in lines),
               sum(1 for ln in lines if "no report" in ln)),
              ([["rewrite_app", "1"]], [["alphalib", "1"]], True, True, 2))
        check("the interpreter the tool runs on: an interpreter given as the "
              "tool, else the python beside the script, else the fallback",
              (er.interpreter_of([sys.executable, "-m", "fixer"], "fb"),
               er.interpreter_of(["no-such-tool-anywhere-xyz"], "fb"),
               er.interpreter_of([str(pathlib.Path(sys.executable).parent
                                      / "pip")], "fb")
               == str(pathlib.Path(sys.executable).parent / "python")),
              (sys.executable, "fb", True))
        check("what is done: a run with an exit code, a version from a wheel; "
              "not a skipped run, not a FAILED version",
              (er.done({"exit": 1}), er.done({"how": "wheel"}),
               er.done({"exit": None, "skipped": "no tree"}),
               er.done({"how": "FAILED", "error": "x"}), er.done(None)),
              (True, True, False, False, False))
        check("pip's one message that means 'no wheel', and the others that "
              "do not",
              (er.no_wheel("ERROR: Could not find a version that satisfies "
                           "the requirement pyyaml==5.4.1\nERROR: No matching "
                           "distribution found for pyyaml==5.4.1"),
               er.no_wheel("ERROR: Could not install packages due to an "
                           "OSError: [Errno 28] No space left"),
               er.no_wheel("WARNING: Retrying (Retry(total=4)) after "
                           "connection broken"),
               er.no_wheel("ERROR: Could not find a version that satisfies "
                           "the requirement redis==7.4.1 (from versions: "
                           "none)\nERROR: No matching distribution found for "
                           "redis==7.4.1")),
              (True, False, False, False))
        calls = er.decisions(
            (root / "rewrite_app" / "alphalib-1.0.0-2.0.0.txt").read_text())
        check("the decision lines, each with its file, line, call, kind and "
              "the rule lines under it",
              (len(calls), sum(1 for c in calls if c["kind"] == "rewritten"),
               sum(1 for c in calls if c["kind"] == "withheld"),
               calls[0]["file"], calls[0]["line"], calls[0]["kind"],
               calls[0]["symbol"], len(calls[0]["rules"])),
              (48, 32, 13, "cases.py", 9, "rewritten",
               "alphalib.core.Client.__init__", 2))
        nested = [c for c in calls
                  if c["file"] == "cases.py" and c["line"] == 30]
        check("two calls on one line at one column are two decisions, told "
              "apart by their use",
              [(c["use"], c["kind"], c["symbol"]) for c in nested],
              [("call", "rewritten", "alphalib.core.Client.__init__"),
               ("method", "rewritten", "alphalib.core.Base.send")])

        print("\n4. THE DRAW")
        per_run = {"A": 10, "B": 2, "C": 0, "D": 5}
        check("the allocation: minimums first, the rest in proportion, "
              "all when the budget covers it",
              (er.allocate(per_run, 8), er.allocate(per_run, 12),
               er.allocate(per_run, 17), er.allocate(per_run, 40),
               er.allocate({k: 4 for k in "ABCDE"}, 10)),
              ({"A": 3, "B": 2, "C": 0, "D": 3},
               {"A": 6, "B": 2, "C": 0, "D": 4},
               {"A": 10, "B": 2, "C": 0, "D": 5},
               {"A": 10, "B": 2, "C": 0, "D": 5},
               {k: 2 for k in "ABCDE"}))
        forty = {f"run{i:02}": 5 for i in range(40)}
        check("one each does not fit: allocate answers None",
              er.allocate(forty, 30), None)
        pool = {"A": [("a.py", i) for i in range(10)],
                "B": [("b.py", 1), ("b.py", 2)], "C": [],
                "D": [("d.py", i) for i in range(5)]}
        d1, how1 = er.draw(pool, 8, seed=1)
        d2, _ = er.draw(pool, 8, seed=1)
        d3, _ = er.draw(pool, 8, seed=2)
        d4, _ = er.draw({**pool, "A": list(reversed(pool["A"]))}, 8, seed=1)
        check("the draw: the allocated number from each run, the same seed "
              "the same calls whatever the list's order, another seed "
              "other calls",
              ({k: len(v) for k, v in d1.items()}, how1, d1 == d2, d1 == d4,
               d1 == d3, d1["B"], all(c in pool["A"] for c in d1["A"])),
              ({"A": 3, "B": 2, "C": 0, "D": 3}, "allocated", True, True,
               False, [("b.py", 1), ("b.py", 2)], True))
        big = {k: [(k, i) for i in range(5)] for k in forty}
        p1, how = er.draw(big, 30, seed=1)
        p2, _ = er.draw(big, 30, seed=1)
        shares = [len(p1[k]) for k in sorted(big)]
        check("the pooled draw: 30 from all the calls as one list, the same "
              "seed the same, spread over many runs and not taken from the "
              "list's head",
              (how, sum(shares), p1 == p2,
               sum(1 for s in shares if s) >= 15,
               shares[:6] == [5] * 6, shares == [1] * 30 + [0] * 10,
               all(c in big[k] for k in p1 for c in p1[k])),
              ("pooled", 30, True, True, False, False, True))

        check("a decision line with no hit line before it is not a "
              "decision",
              er.decisions("                                    -> rewritten: "
                           "x()\n"), [])

        print("\n5. THE OLD VERSION'S DIRECTORY FROM AN SDIST, NOT BUILT")
        sd = tmp / "sd" / "pkg-1.0"
        (sd / "lib" / "yam").mkdir(parents=True)
        (sd / "lib" / "yam" / "__init__.py").write_text("python2 = True\n")
        (sd / "lib3" / "yam").mkdir(parents=True)
        (sd / "lib3" / "yam" / "__init__.py").write_text("python3 = True\n")
        (sd / "setup.py").write_text("raise SystemExit('never run')\n")
        dest = tmp / "old" / "PyYam-1.0"
        found = er.place_unpacked(sd, "PyYam", "1.0", dest)
        meta = (dest / "pyyam-1.0.dist-info" / "METADATA").read_text()
        installed = find_installed("PyYam", dest)
        check("the package directory found under the sdist (lib3 before "
              "lib, as pyyaml 5.4.1 lays it out) and placed beside a "
              "hand-written dist-info that fixer/env.py reads back; nothing "
              "run",
              (found, (dest / "yam" / "__init__.py").read_text(),
               "Name: PyYam" in meta and "Version: 1.0" in meta,
               (dest / "pyyam-1.0.dist-info" / "top_level.txt")
               .read_text().strip(),
               (installed.version, installed.import_roots)),
              (["yam"], "python3 = True\n", True, "yam", ("1.0", ("yam",))))
        single = tmp / "sd" / "te-4.0"
        (single / "src").mkdir(parents=True)
        (single / "src" / "typing_extensions.py").write_text("x = 1\n")
        (single / "setup.py").write_text("raise SystemExit('never run')\n")
        (single / "src" / "tools").mkdir()
        (single / "src" / "tools" / "__init__.py").write_text("")
        dest2 = tmp / "old" / "typing-extensions-4.0"
        found2 = er.place_unpacked(single, "typing-extensions", "4.0", dest2)
        check("a single-module distribution is placed as its module; a "
              "package named like a skipped directory is kept under src/",
              (found2, (dest2 / "typing_extensions.py").is_file(),
               find_installed("typing_extensions", dest2).import_roots),
              (["tools", "typing_extensions"], True,
               ("tools", "typing_extensions")))
        six = tmp / "sd" / "six-1.16.0"
        six.mkdir(parents=True)
        (six / "six.py").write_text("x = 1\n")
        (six / "setup.py").write_text("raise SystemExit('never run')\n")
        (six / "test_six.py").write_text("")
        check("a single module at the sdist's root is placed, setup.py and "
              "the tests left aside",
              er.place_unpacked(six, "six", "1.16.0", tmp / "old" / "six"),
              ["six"])
        blob = tmp / "blob.bin"
        blob.write_bytes(b"abc")
        right = ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61"
                 "f20015ad")
        try:
            er.verify(blob, "0" * 64)
            wrong = None
        except RuntimeError as e:
            wrong = str(e)
        check("a download whose sha256 is not PyPI's is deleted and refused; "
              "the right one passes with its size",
              (wrong is not None and "sha256" in wrong, blob.exists(),
               (blob.write_bytes(b"abc"), er.verify(blob, right))[1]),
              (True, False, 3))

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The harness plans, runs, parses and draws as")
    print("§38.18 says, and the test touched no network.")


if __name__ == "__main__":
    main()
