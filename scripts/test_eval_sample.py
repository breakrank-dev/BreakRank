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
removed), LIMIT, RATE, ping, compat, _internal, the class rows.

The pairs:
  appa   alphalib.Client, alphalib.helper, alphalib.util.LIMIT
           -> Client.__init__ (the class path), helper: 2
  appb   alphalib.core.Client.__init__, alphalib.util.spread,
         alphalib.opts.fetch, alphalib.opts.Session, alphalib.opts.Session.get
           -> 5
  appc   alphalib.util.LIMIT, alphalib.util.ping -> 0, not listed
  alphalib  alphalib.Client -> self-use, dropped
  appd   alphalib.Base.send, alphalib.connect -> 2
  appe   alphalib.opts.shape, alphalib.opts.fetch -> 2
  appf   alphalib.Client.from_url -> 1
  appg   alphalib.Client.from_url, alphalib.Client.parse, alphalib.spread -> 3
  aa     alphalib.opts.ship, alphalib.opts.emit -> 2, listed first in the
         file and shorter than the others, so that a tie broken any way
         but by name shows
Ranked: appb 5, appg 3, aa 2, appa 2, appd 2, appe 2, appf 1 (ties by
name). With --top 3: the sample appb, appg, aa; the reserves appa,
appd, appe, appf.
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
    changes, extra_changes, r2_changes, rewrite_changes, write_bundle)

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name, got, want):
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


PAIRS = [
    ("aa", "alphalib.opts.ship"), ("aa", "alphalib.opts.emit"),
    ("appa", "alphalib.Client"), ("appa", "alphalib.helper"),
    ("appa", "alphalib.util.LIMIT"),
    ("appb", "alphalib.core.Client.__init__"), ("appb", "alphalib.util.spread"),
    ("appb", "alphalib.opts.fetch"), ("appb", "alphalib.opts.Session"),
    ("appb", "alphalib.opts.Session.get"),
    ("appc", "alphalib.util.LIMIT"), ("appc", "alphalib.util.ping"),
    ("alphalib", "alphalib.Client"),
    ("appd", "alphalib.Base.send"), ("appd", "alphalib.connect"),
    ("appe", "alphalib.opts.shape"), ("appe", "alphalib.opts.fetch"),
    ("appf", "alphalib.Client.from_url"),
    ("appg", "alphalib.Client.from_url"), ("appg", "alphalib.Client.parse"),
    ("appg", "alphalib.spread"),
]


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
              (upgrades[("alphalib", "alphalib.core.Client.__init__")],
               sorted(rules[("alphalib", "alphalib.core.Client.__init__")]),
               sorted(rules[("alphalib", "alphalib.core.connect")]),
               sorted(rules[("alphalib", "alphalib.opts.wrap")]),
               sorted(rules[("alphalib", "alphalib.opts.fetch")])),
              ({("1.0.0", "2.0.0")}, ["R1", "R2"], ["R1"], ["R1", "R2"],
               ["R2"]))
        check("the class path reaches the constructor; a flag-only "
              "function and a constant reach nothing",
              (sorted(by_path["alphalib.Client"]),
               by_path.get("alphalib.util.Pool.acquire"),
               by_path.get("alphalib.util.LIMIT")),
              ([("alphalib", "alphalib.core.Client.__init__")], None, None))

        print("\n3. THE PAIRS, RANKED, SELF-USE DROPPED")
        pairs = tmp / "pairs.csv"
        with pairs.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scanner", "symbol"])
            w.writerows(PAIRS)
        read, digest = es.read_pairs(pairs)
        check("the pairs read back, with a digest", (len(read), len(digest)),
              (21, 64))
        ranked = es.rank(read, by_path)
        check("ranked by distinct functions, ties by name; appc with none "
              "and alphalib's self-use are out",
              [(s, len(k)) for s, k in ranked],
              [("appb", 5), ("appg", 3), ("aa", 2), ("appa", 2),
               ("appd", 2), ("appe", 2), ("appf", 1)])
        check("appa's functions: the constructor through the class path, "
              "and helper",
              sorted(f for _, f in dict(ranked)["appa"]),
              ["alphalib.core.Client.__init__", "alphalib.core.helper"])
        lines = es.describe(ranked, upgrades, rules, top=3, show=25)
        check("the table marks the sample and the reserves",
              ([ln.split()[-3:] for ln in lines[1:8]]),
              [["appb", "<-", "sample"], ["appg", "<-", "sample"],
               ["aa", "<-", "sample"], ["appa", "<-", "reserve"],
               ["appd", "<-", "reserve"], ["appe", "<-", "reserve"],
               ["appf", "<-", "reserve"]])
        check("the sample's detail names the package, the upgrade and the "
              "functions with their rules",
              ("   1. appb: 5 function(s)" in lines,
               "      alphalib: 5 function(s) in 1 upgrade(s): 1.0.0 -> 2.0.0"
               in lines,
               "          alphalib.core.Client.__init__   [R1+R2]" in lines,
               "          alphalib.opts.Session.get   [R2]" in lines),
              (True, True, True, True))

        print("\n4. THE COMMAND LINE")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = es.main(["--pairs", str(pairs), "--bundle", str(bundle),
                            "--out", str(tmp / "sample.csv"), "--top", "3"])
        text = out.getvalue()
        check("exit 0; the header counts the pairs and the functions",
              (code, "21 pairs, sha256" in text,
               "14 rewritable function(s) in 1 package(s)" in text,
               "7 scanning package(s) use at least one; 17 (package, "
               "function) uses in all" in text),
              (1 - 1, True, True, True))
        with (tmp / "sample.csv").open() as f:
            rows_out = list(csv.DictReader(f))
        check("the csv: one row per ranked package, with the detail",
              ([(r["rank"], r["scanner"], r["functions"]) for r in rows_out],
               rows_out[0]["detail"].split(";")[0]),
              ([("1", "appb", "5"), ("2", "appg", "3"), ("3", "aa", "2"),
                ("4", "appa", "2"), ("5", "appd", "2"), ("6", "appe", "2"),
                ("7", "appf", "1")],
               "alphalib:alphalib.core.Client.__init__"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = es.main(["--pairs", str(tmp / "missing.csv"),
                            "--bundle", str(bundle)])
        check("a missing pairs file: exit 2 and where it comes from",
              (code, "run_usage.py" in err.getvalue()), (2, True))

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The sample is the rule's, from the pairs and")
    print("the bundle, with self-use dropped and the reserves named.")


if __name__ == "__main__":
    main()
