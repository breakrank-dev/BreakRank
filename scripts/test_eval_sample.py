"""
Does scripts/eval_sample.py pick the sample by the rule of NOTES §38.16,
from the pairs and the bundle, and nothing else?

    python scripts/test_eval_sample.py

No network, no real data: the bundle is the fixture's (the forty-five
rows of scripts/test_fixer_locate.py) and the pairs are written here.
Expected, written BEFORE the code ran. The fixture's rewritable
functions, fourteen: Client.__init__ (port, db moved; retries default),
Base.send (timeout keyword-only), helper (b moved; d default), connect
(port moved, beside a removal and a kind change), spread (first moved),
Client.from_url (retries moved), Client.parse (strict keyword-only),
fetch (timeout, retries, verify, pool, tags defaults), shape (b default;
c required), Session.__init__ (protocol, keepalive), Session.get
(default; ghost), ship (compress default), emit (items moved), wrap (y
required; z default and moved). Not rewritable: Pool.acquire (a
parameter added as required, a flag alone), compute (a parameter
removed), LIMIT, RATE, ping, compat, _internal, the class rows. No path
of the fixture reaches two defining paths.

The pairs, 26 rows, 24 distinct:
  appa   alphalib.Client, alphalib.helper, alphalib.util.LIMIT
           -> Client.__init__ (the class path), helper: 2
  appb   alphalib.core.Client.__init__, alphalib.util.spread,
         alphalib.opts.fetch, alphalib.opts.Session, alphalib.opts.Session.get
           -> 5
  appc   alphalib.util.LIMIT, alphalib.util.ping -> 0, not listed
  alphalib  alphalib.Client -> self-use, dropped
  appd   alphalib.Base.send, alphalib.connect -> 2
  appe   alphalib.opts.shape, alphalib.opts.fetch -> 2
  appf   alphalib.Client.from_url -> 1; its row written twice, the second
         spelled APPF: the same pair by PEP 503 name, counted once
  appg   alphalib.Client.from_url, alphalib.Client.parse, alphalib.spread -> 3
  appb-slim  alphalib.opts.fetch, alphalib.opts.Session -> 2; a second
         distribution of appb's repository (the test's same-source table
         says so), using two of appb's five
  AlphaLib  alphalib.helper -> self-use by PEP 503 name, dropped
  " appb"  alphalib.opts.fetch -> appb's, by PEP 503 name after a strip:
         merged into appb, which keeps the spelling seen first
  aa     alphalib.opts.ship, alphalib.opts.emit -> 2, written LAST in the
         file and first by name, so that a tie broken by file order shows
Ranked: appb 5, appg 3, aa 2, appa 2, appb-slim 2, appd 2, appe 2,
appf 1 (ties by name): 8 packages, 19 uses. With --top 3 and appb-slim
listed as appb's second distribution (spelled AppB_Slim in the table,
the same name by PEP 503): the sample appb, appg, aa; appb-slim skipped
and the positions closed up; the reserves appa, appd, appe, appf with
the default five, and appa, appd, appe with three (appf nothing, which
shows the closing up); the last sampled, aa, ties on 2 with appa, the
first reserve, and the run says so. A table that lists appg as appb's
second distribution is refused: appg uses Client.from_url and
Client.parse, which appb does not (spread, both do). A table whose first
distribution is not ranked keeps the second as the repository's entry
and says so in a note. A name listed both as a second distribution and
as a first (a pair both ways, a self-pair, a chain) is refused. The
skipped mark carries the first's own spelling. With --top 5 the fifth
sampled is appd at rank 6, the skipped row inside the sample's span;
with --top 20 all seven are the sample and the header says seven; with
--show 2 the table still runs through the last reserve. A short row in
the pairs file is refused with its line number; a tie is read past a
skipped row. A standing second distribution (its first not ranked)
carries its note into the csv; outside the fifteen the note says so;
the first is spelled as the table writes it. 30 checks.
"""

import contextlib
import csv
import io
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scripts.eval_sample as es  # noqa: E402
from scripts.test_fixer_locate import (  # noqa: E402
    change, changes, extra_changes, r2_changes, rewrite_changes,
    write_bundle)

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name, got, want):
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


PAIRS = [
    ("appa", "alphalib.Client"), ("appa", "alphalib.helper"),
    ("appa", "alphalib.util.LIMIT"),
    ("appb", "alphalib.core.Client.__init__"),
    ("appb", "alphalib.util.spread"),
    ("appb", "alphalib.opts.fetch"), ("appb", "alphalib.opts.Session"),
    ("appb", "alphalib.opts.Session.get"),
    ("appc", "alphalib.util.LIMIT"), ("appc", "alphalib.util.ping"),
    ("alphalib", "alphalib.Client"),
    ("appd", "alphalib.Base.send"), ("appd", "alphalib.connect"),
    ("appe", "alphalib.opts.shape"), ("appe", "alphalib.opts.fetch"),
    ("appf", "alphalib.Client.from_url"),
    ("appg", "alphalib.Client.from_url"), ("appg", "alphalib.Client.parse"),
    ("appg", "alphalib.spread"),
    ("appb-slim", "alphalib.opts.fetch"),
    ("appb-slim", "alphalib.opts.Session"),
    ("AlphaLib", "alphalib.helper"),
    ("APPF", "alphalib.Client.from_url"),
    (" appb", "alphalib.opts.fetch"),
    ("aa", "alphalib.opts.ship"), ("aa", "alphalib.opts.emit"),
]
SAME_SOURCE = {"AppB_Slim": "appb"}  # PEP 503: the same name as appb-slim

HELPER = ("alphalib", "alphalib.core.helper")
INIT = ("alphalib", "alphalib.core.Client.__init__")


def main() -> None:
    print("\n1. WHICH ROWS ARE REWRITABLE")
    check("R1's kinds: moved, and a kind change to keyword-only; not to "
          "positional-only",
          (es.rewritable("PARAMETER_MOVED", "f(x): Positional parameter was "
                         "moved", None, None),
           es.rewritable("PARAMETER_CHANGED_KIND", "f(x): Parameter kind was "
                         "changed: positional or keyword -> keyword-only",
                         None, None),
           es.rewritable("PARAMETER_CHANGED_KIND", "f(x): Parameter kind was "
                         "changed: positional or keyword -> positional-only",
                         None, None)),
          ("R1", "R1", None))
    check("R2's kinds: a plain old default that changed, a parameter made "
          "required; not a name, not a container, not a spelling-only "
          "change, not a row without values",
          (es.rewritable("PARAMETER_CHANGED_DEFAULT", "", "None", "X"),
           es.rewritable("PARAMETER_CHANGED_REQUIRED", "", None, None),
           es.rewritable("PARAMETER_CHANGED_DEFAULT", "", "DEFAULT", "2"),
           es.rewritable("PARAMETER_CHANGED_DEFAULT", "", "[]", "None"),
           es.rewritable("PARAMETER_CHANGED_DEFAULT", "", "'a'", '"a"'),
           es.rewritable("PARAMETER_CHANGED_DEFAULT", "", None, None)),
          ("R2", "R2", None, None, None, None))
    check("the flag-only kinds are not rewritable",
          tuple(es.rewritable(k, "", None, None) for k in
                ("PARAMETER_REMOVED", "PARAMETER_ADDED_REQUIRED",
                 "OBJECT_REMOVED", "ATTRIBUTE_CHANGED_VALUE",
                 "CLASS_REMOVED_BASE")),
          (None,) * 5)
    check("the paths of a function: defining, exports; a constructor adds "
          "the class's paths",
          (es.paths_of("alphalib.core.Client.__init__",
                       "alphalib.Client.__init__"),
           es.paths_of("alphalib.core.helper", "alphalib.helper"),
           es.paths_of("alphalib.opts.fetch", "")),
          ({"alphalib.core.Client.__init__", "alphalib.Client.__init__",
            "alphalib.core.Client", "alphalib.Client"},
           {"alphalib.core.helper", "alphalib.helper"},
           {"alphalib.opts.fetch"}))

    print("\n2. THE FIXTURE'S BUNDLE: FOURTEEN REWRITABLE FUNCTIONS")
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        bundle = tmp / "b.sqlite"
        rows = changes() + extra_changes() + rewrite_changes() + r2_changes()
        write_bundle(bundle, rows)
        by_path, upgrades, rules = es.load_functions(bundle)
        functions = sorted({k for keys in by_path.values() for k in keys})
        check("the fourteen, by defining path",
              [f for _, f in functions],
              ["alphalib.core.Base.send", "alphalib.core.Client.__init__",
               "alphalib.core.Client.from_url", "alphalib.core.Client.parse",
               "alphalib.core.connect", "alphalib.core.helper",
               "alphalib.opts.Session.__init__", "alphalib.opts.Session.get",
               "alphalib.opts.emit", "alphalib.opts.fetch",
               "alphalib.opts.shape", "alphalib.opts.ship",
               "alphalib.opts.wrap", "alphalib.util.spread"])
        check("each with its upgrade and its rules",
              (upgrades[INIT], sorted(rules[INIT]),
               sorted(rules[("alphalib", "alphalib.core.connect")]),
               sorted(rules[("alphalib", "alphalib.opts.wrap")]),
               sorted(rules[("alphalib", "alphalib.opts.fetch")])),
              ({("1.0.0", "2.0.0")}, ["R1", "R2"], ["R1"], ["R1", "R2"],
               ["R2"]))
        check("the class path reaches the constructor; a flag-only "
              "function and a constant reach nothing; no path of the "
              "fixture reaches two defining paths",
              (sorted(by_path["alphalib.Client"]),
               by_path.get("alphalib.util.Pool.acquire"),
               by_path.get("alphalib.util.LIMIT"),
               es.shared_paths(by_path)),
              ([INIT], None, None, []))
        moved = tmp / "moved.sqlite"
        write_bundle(moved, [
            change(901, "alphalib.core.helper", "PARAMETER_MOVED", "b",
                   "alphalib.helper"),
            change(902, "alphalib.util.helper", "PARAMETER_MOVED", "b",
                   "alphalib.helper")])
        m_paths, _, _ = es.load_functions(moved)
        check("a function that moved modules counts once per defining "
              "path, and the shared export path is reported",
              (es.shared_paths(m_paths),
               [(s, len(k)) for s, k in
                es.rank([("appa", "alphalib.helper")], m_paths)]),
              (["alphalib.helper"], [("appa", 2)]))

        print("\n3. THE PAIRS, RANKED, SELF-USE DROPPED")
        pairs = tmp / "pairs.csv"
        with pairs.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scanner", "symbol"])
            w.writerows(PAIRS)
        read, digest = es.read_pairs(pairs)
        check("the pairs read back, stripped, with a digest; 24 distinct by "
              "PEP 503 name",
              (len(read), len(digest), es.distinct(read),
               read[23]),
              (26, 64, 24, ("appb", "alphalib.opts.fetch")))
        ranked = es.rank(read, by_path)
        check("ranked by distinct functions, ties by name; appc with none, "
              "alphalib's and AlphaLib's self-use out; appf's doubled row "
              "and appb's second spelling counted once",
              [(s, len(k)) for s, k in ranked],
              [("appb", 5), ("appg", 3), ("aa", 2), ("appa", 2),
               ("appb-slim", 2), ("appd", 2), ("appe", 2), ("appf", 1)])
        check("appa's functions: the constructor through the class path, "
              "and helper",
              sorted(f for _, f in dict(ranked)["appa"]),
              ["alphalib.core.Client.__init__", "alphalib.core.helper"])
        three, notes3 = es.select(ranked, top=3, reserves=3,
                                  same_source=SAME_SOURCE)
        check("a second distribution of a repository is skipped and the "
              "positions close up: with three reserves, appe is the third "
              "and appf is nothing; no note",
              ([(s, three[s]) for s, _ in ranked], notes3),
              ([("appb", "sample"), ("appg", "sample"), ("aa", "sample"),
                ("appa", "reserve"),
                ("appb-slim", "same source as appb, skipped"),
                ("appd", "reserve"), ("appe", "reserve"), ("appf", "")],
               {}))
        roles, notes = es.select(ranked, top=3, same_source=SAME_SOURCE)

        def refused(table, top=3):
            try:
                es.select(ranked, top=top, same_source=table)
            except es.SampleError as e:
                return str(e)
            return None

        wrong = refused({"appg": "appb"})
        check("a table the data contradicts is refused, naming the "
              "functions and the two causes",
              (wrong is not None and "appg" in wrong and "appb" in wrong
               and "alphalib.core.Client.from_url" in wrong
               and "alphalib.core.Client.parse" in wrong
               and "alphalib.util.spread" not in wrong
               and "different versions" in wrong),
              True)
        kept, kept_notes = es.select(ranked, top=3, same_source={"aa": "Zzz"})
        out_of, out_notes = es.select(ranked, top=1, reserves=1,
                                      same_source={"appf": "Zzz"})
        check("a first distribution that is not ranked: the second stands "
              "for the repository, with a note spelling the first as the "
              "table does; outside the fifteen the note says so",
              (kept["aa"], list(kept_notes),
               "Zzz" in kept_notes.get("aa", "")
               and "not ranked" in kept_notes.get("aa", "")
               and "stands for the repository" in kept_notes.get("aa", ""),
               out_of["appf"], "outside the sample and the reserves"
               in out_notes.get("appf", "")),
              ("sample", ["aa"], True, "", True))
        check("a name listed both as a second distribution and as a first "
              "is refused: a pair both ways, a self-pair, a chain",
              tuple("both" in (refused(t) or "") for t in
                    ({"appb-slim": "appb", "appb": "appb-slim"},
                     {"appb": "appb"},
                     {"appb-slim": "appb", "appb": "appg"})),
              (True, True, True))
        odd = [("Foo.Bar", {HELPER, INIT}), ("foo-bar-slim", {HELPER})]
        odd_roles, _ = es.select(odd, top=1, same_source={"foo-bar-slim":
                                                          "Foo.Bar"})
        check("the skipped mark carries the first's own spelling",
              odd_roles["foo-bar-slim"], "same source as Foo.Bar, skipped")
        lines = es.describe(ranked, upgrades, rules, roles, top=3, show=25)
        check("the table marks the sample, the reserves and the skipped "
              "distribution",
              [(ln.split()[4], ln.split("<- ")[1] if "<- " in ln else "")
               for ln in lines[1:9]],
              [("appb", "sample"), ("appg", "sample"), ("aa", "sample"),
               ("appa", "reserve"),
               ("appb-slim", "same source as appb, skipped"),
               ("appd", "reserve"), ("appe", "reserve"),
               ("appf", "reserve")])
        check("a tie at the sample's cut is said; none at the reserves' "
              "(appf is last)",
              ([ln for ln in lines if ln.startswith("the last")]),
              ["the last sampled, aa, ties on 2 with appa, the first "
               "reserve: by name"])
        four, _ = es.select(ranked, top=4, same_source=SAME_SOURCE)
        check("the tie is read past a skipped row: with --top 4, appa ties "
              "with appd, not with appb-slim",
              [ln for ln in es.describe(ranked, upgrades, rules, four,
                                        top=4, show=25)
               if ln.startswith("the last")],
              ["the last sampled, appa, ties on 2 with appd, the first "
               "reserve: by name"])
        check("the sample's detail names the package with its rank, the "
              "upgrade and the functions with their rules",
              ("   1. appb, rank 1: 5 function(s)" in lines,
               "   3. aa, rank 3: 2 function(s)" in lines,
               "      alphalib: 5 function(s) in 1 upgrade(s): 1.0.0 -> 2.0.0"
               in lines,
               "          alphalib.core.Client.__init__   [R1+R2]" in lines,
               "          alphalib.opts.Session.get   [R2]" in lines,
               any(ln.startswith("   4.") for ln in lines)),
              (True, True, True, True, True, False))
        five, _ = es.select(ranked, top=5, same_source=SAME_SOURCE)
        wide = es.describe(ranked, upgrades, rules, five, top=5, show=25)
        check("a skipped distribution inside the sample's span: the fifth "
              "sampled is appd, at rank 6",
              ("   5. appd, rank 6: 2 function(s)" in wide,
               any(ln.startswith("   6.") for ln in wide),
               [s for s, _ in ranked if five[s] == "sample"]),
              (True, False, ["appb", "appg", "aa", "appa", "appd"]))
        twenty, _ = es.select(ranked, top=20, same_source=SAME_SOURCE)
        head = es.describe(ranked, upgrades, rules, twenty, top=20, show=25)
        check("--top beyond the ranking: all seven are the sample and the "
              "header says seven",
              (sorted(s for s, _ in ranked if twenty[s] == "sample"),
               [ln for ln in head if ln.startswith("the sample, the")]),
              (["aa", "appa", "appb", "appd", "appe", "appf", "appg"],
               ["the sample, the 7 with the most rewritable functions "
                "(ties by name; one entry per repository):"]))
        short = es.describe(ranked, upgrades, rules, roles, top=3, show=2)
        check("--show below the last reserve: the table runs through it",
              (short.index(""), short[8].split()[4]), (10, "appf"))
        ups = {HELPER: {("2.9.2", "2.10.0"), ("2.10.0", "2.11.0"),
                        ("2.8.0", "2.9.0")}}
        order = es.describe([("p", {HELPER})], ups, {HELPER: {"R1"}},
                            {"p": "sample"}, top=1, show=5)
        check("the detail's upgrades in version order, not text order",
              [ln for ln in order if ln.startswith("      alphalib:")],
              ["      alphalib: 1 function(s) in 3 upgrade(s): "
               "2.8.0 -> 2.9.0, 2.9.2 -> 2.10.0, 2.10.0 -> 2.11.0"])

        print("\n4. THE COMMAND LINE")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = es.main(["--pairs", str(pairs), "--bundle", str(bundle),
                            "--out", str(tmp / "sample.csv"), "--top", "3"],
                           same_source=SAME_SOURCE)
        text = out.getvalue()
        check("exit 0; the header counts the rows, the distinct pairs, the "
              "full digest, the functions and the shared paths",
              (code,
               f"{pairs}: 26 row(s), 24 distinct pair(s), sha256 {digest}"
               in text,
               f"{bundle}: 14 rewritable function(s) in 1 package(s); 0 "
               "path(s) reach more than one defining path" in text,
               "8 scanning package(s) use at least one; 19 (package, "
               "function) uses in all" in text),
              (0, True, True, True))
        with (tmp / "sample.csv").open() as f:
            rows_out = list(csv.DictReader(f))
        check("the csv: one row per ranked package, with its role, an "
              "empty note and the detail",
              ([(r["rank"], r["scanner"], r["functions"], r["role"])
                for r in rows_out],
               {r["note"] for r in rows_out},
               rows_out[0]["detail"].split(";")[0]),
              ([("1", "appb", "5", "sample"), ("2", "appg", "3", "sample"),
                ("3", "aa", "2", "sample"), ("4", "appa", "2", "reserve"),
                ("5", "appb-slim", "2", "same source as appb, skipped"),
                ("6", "appd", "2", "reserve"), ("7", "appe", "2", "reserve"),
                ("8", "appf", "1", "reserve")],
               {""},
               "alphalib:alphalib.core.Client.__init__"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = es.main(["--pairs", str(pairs), "--bundle", str(bundle),
                            "--out", str(tmp / "stands.csv"), "--top", "3"],
                           same_source={"aa": "Zzz"})
        with (tmp / "stands.csv").open() as f:
            stands = {r["scanner"]: (r["role"], r["note"])
                      for r in csv.DictReader(f)}
        check("a standing second distribution: the note under the table "
              "and in the csv's note column, its role plain",
              (code, [ln for ln in out.getvalue().splitlines()
                      if ln.startswith("note: ")][0][:46],
               stands["aa"][0], "Zzz" in stands["aa"][1],
               stands["appb"][1]),
              (0, "note: aa is listed as a second distribution of", "sample",
               True, ""))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = es.main(["--pairs", str(tmp / "missing.csv"),
                            "--bundle", str(bundle)])
        check("a missing pairs file: exit 2 and where it comes from",
              (code, "run_usage.py" in err.getvalue()), (2, True))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            code = es.main(["--pairs", str(pairs), "--bundle", str(bundle),
                            "--out", str(tmp / "none.csv")],
                           same_source={"appg": "appb"})
        check("a wrong same-source table: exit 2, the message, and no csv",
              (code, "appg" in err.getvalue(),
               (tmp / "none.csv").exists()),
              (2, True, False))
        bad = tmp / "short.csv"
        bad.write_text("scanner,symbol\nappz\nappa,alphalib.helper\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            code = es.main(["--pairs", str(bad), "--bundle", str(bundle),
                            "--out", str(tmp / "none.csv")])
        check("a short row in the pairs file: exit 2 with its line number",
              (code, f"{bad}:2:" in err.getvalue(),
               (tmp / "none.csv").exists()),
              (2, True, False))

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The sample is the rule's, from the pairs and")
    print("the bundle, with self-use dropped and the reserves named.")


if __name__ == "__main__":
    main()
