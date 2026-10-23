"""
Does R1 rewrite exactly the calls §38.4 says, with the names the old
signature gives, and withhold with the reason everywhere else?

    python scripts/test_fixer_rewrite.py

No database, no network, nothing imported from the fixtures. The project
is scripts/fixtures/rewrite_app/cases.py, one call per line against the
pretend package alphalib 1.0.0 (scripts/fixtures/site/), with the twenty
changes of scripts/test_fixer_locate.py (its twelve, the review's three,
and five for these cases). Client.__init__(self, host, port=6379, db=0,
retries=3) has port and db MOVED and retries re-defaulted; Base.send(self,
payload, timeout=10) has timeout made keyword-only; helper(a, b, /, c, *,
d=1) has b MOVED, a positional-only parameter; connect(host, port=1,
timeout=5) has timeout REMOVED and port both MOVED and made
positional-only; spread(first, *rest, flag=False) has first MOVED and
rest's kind changed in a way R1 does not cover; the classmethod
Client.from_url(cls, url, retries=3) has retries MOVED and a timeout ADDED
as required; the staticmethod Client.parse(url, strict=False) has strict
made keyword-only.
Expected, written BEFORE the code ran (line: call -> decision):

   9  Client("localhost", 6380, 2)
        rewrite  Client("localhost", port=6380, db=2)
  10  Client("localhost", 6380)      rewrite  Client("localhost", port=6380)
  11  Client(..., port=, db=)        nothing  already passed by name
  12  Client("localhost")            nothing  the moved parameters not passed
  13  Client("h", 1, 2, 3)
        rewrite  Client("h", port=1, db=2, retries=3): retries is R2's
        row, but it sits in the run, so it is named
  14  Client(*d.args)                flag     unpacking: positions unknown
  15  Client("h", 1, db=2)           rewrite  Client("h", port=1, db=2)
  16  Client("h", 1, 2, 3, 4)        flag     5 positional, 4 named: wrong
  17  a.send(b"x", 5)                rewrite  a.send(b"x", timeout=5), self
                                              offset
  18  Base.send(a, b"x", 5)          rewrite  Base.send(a, b"x", timeout=5),
                                              no offset
  19  a.send(b"x")                   nothing  timeout not passed
  20  helper(1, 2, 3)                flag     b was positional-only in the
                                              old version
  21  connect("h", 2, timeout=9)     flag     port became positional-only,
                                              timeout removed; would be
                                              connect("h", port=2, timeout=9)
  22  Child("h", 1, 2)               rewrite  Child("h", port=1, db=2): the
                                              inherited __init__, Client's
                                              rows and signature
  23  Client.from_url("u", 5)        flag     timeout added as required;
                                              would be Client.from_url("u",
                                              retries=5): cls offset, from
                                              griffe's classmethod label
  24  alphalib.Client("h", 1)        rewrite  alphalib.Client("h", port=1)
  25  Client("h", 1,  # the port     rewrite  Client("h", port=1,  # the port
             2)                                          db=2), comment kept
  27  spread("a", "b", "c", True)    flag     4 positional, 1 named: the
                                              rest went to *args; and rest's
                                              kind change is not R1's case
  28  Client.parse("u", True)        rewrite  Client.parse("u", strict=True)
                                              no offset: staticmethod
  29  Client("h", 1, port=2)         flag     port by position and keyword
  30  Client("h", 1).send(b"t", 5)   rewrite  two calls at one column: the
                                              constructor and the method on
                                              its result, both named:
                                              Client("h", port=1).send(b"t",
                                              timeout=5)
  31  Client("h", 1, 2, 3, retries=9) flag    retries, not moved but in the
                                              run, by position and keyword
  32  helper(1, 2, 3, 4)             flag     4 positional, 3 named (d is
                                              keyword-only): already wrong
  33  Client("h", 1)("x")            rewrite  Client("h", port=1)("x"): the
                                              call on the result is not the
                                              package's, the inner one is
  34  Client("h", 1)("x").send(...)  rewrite  the constructor only; the
                                              .send is on what calling the
                                              instance returned: unverified
  35  r = Client("h", 1)("x")        rewrite  the constructor; r is unknown
  36  r.send(b"t", 6)                         unverified, no decision

  15 rewrites in one file (15 source lines, the comment case being two
  and line 30 carrying two), 9 withheld, 3 need nothing; the two calls on
  what an instance returned (34, 36) are unverified. Without the old
  version the two calls on the instance `a` and the `.send` on line 30
  are unverified, so not R1's, and Child(...) is no hit; of the 23 left,
  the unpacking and the all-by-name case are decided without the
  signature and the other 21 are withheld with "the old signature could
  not be read". Idempotence: the diff applied to a copy, locate and R1
  run again on the copy: no rewrite, the fifteen former rewrites
  "already passed by name" (18 need nothing with the three), the nine
  withheld as before, and an empty diff. The diff applies with
  `patch -p1`, also for a file whose last line has no newline; a
  byte-order mark survives `--apply`; one call meeting two defining paths
  gets one decision from both their rows.
"""

import contextlib
import io
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fixer.cli as cli  # noqa: E402
import fixer.locate as loc  # noqa: E402
import fixer.report as rep  # noqa: E402
import fixer.oldapi as oldapi  # noqa: E402
import fixer.rewrite as rw  # noqa: E402
from scripts.test_fixer_locate import (  # noqa: E402
    SITE, change, changes, extra_changes, rewrite_changes, write_bundle)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE / "fixtures" / "rewrite_app"
CASES = "cases.py"

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


def all_changes():
    return changes() + extra_changes() + rewrite_changes()


def decisions(plans):
    """(line, action, names, would_be) per plan, in line order."""
    return sorted((p.hit.line, p.action, p.names, p.would_be) for p in plans)


EXPECTED = [
    (9, "rewrite", ("port", "db"), 'Client("localhost", port=6380, db=2)'),
    (10, "rewrite", ("port",), 'Client("localhost", port=6380)'),
    (11, "nothing", (), None),
    (12, "nothing", (), None),
    (13, "rewrite", ("port", "db", "retries"),
     'Client("h", port=1, db=2, retries=3)'),
    (14, "flag", (), None),
    (15, "rewrite", ("port",), 'Client("h", port=1, db=2)'),
    (16, "flag", (), None),
    (17, "rewrite", ("timeout",), 'a.send(b"x", timeout=5)'),
    (18, "rewrite", ("timeout",), 'Base.send(a, b"x", timeout=5)'),
    (19, "nothing", (), None),
    (20, "flag", (), None),
    (21, "flag", ("port",), 'connect("h", port=2, timeout=9)'),
    (22, "rewrite", ("port", "db"), 'Child("h", port=1, db=2)'),
    (23, "flag", ("retries",), 'Client.from_url("u", retries=5)'),
    (24, "rewrite", ("port",), 'alphalib.Client("h", port=1)'),
    (25, "rewrite", ("port", "db"),
     'Client("h", port=1,  # the port\n           db=2)'),
    (27, "flag", (), None),
    (28, "rewrite", ("strict",), 'Client.parse("u", strict=True)'),
    (29, "flag", (), None),
    (30, "rewrite", ("port",), 'Client("h", port=1)'),
    (30, "rewrite", ("timeout",), 'Client("h", port=1).send(b"t", timeout=5)'),
    (31, "flag", (), None),
    (32, "flag", (), None),
    (33, "rewrite", ("port",), 'Client("h", port=1)'),
    (34, "rewrite", ("port",), 'Client("h", port=1)'),
    (35, "rewrite", ("port",), 'Client("h", port=1)'),
]


def case_parsing() -> None:
    print("\n1. GRIFFE'S WORDS FOR A KIND CHANGE, AND THE SELF OFFSET")
    check("the old and new kinds out of the message",
          (rw.kind_change("f(x): Parameter kind was changed: positional or "
                          "keyword -> keyword-only"),
           rw.kind_change("f(x): Parameter kind was changed: positional or "
                          "keyword -> positional-only"),
           rw.kind_change("f(x): Positional parameter was moved")),
          (("positional or keyword", "keyword-only"),
           ("positional or keyword", "positional-only"), None))
    mk = lambda use, receiver, explicit: loc.Hit(  # noqa: E731
        "f.py", 1, 1, "", "p", (1,), "resolved", use, receiver, explicit)
    check("the offset: a constructor and an instance call skip self, a "
          "classmethod on the class skips cls, a function, an explicit-self "
          "call and a staticmethod skip nothing",
          (rw.offset(mk("call", "alphalib.core.Client", False), set()),
           rw.offset(mk("method", "alphalib.core.Base", False), set()),
           rw.offset(mk("method", "alphalib.core.Client", False),
                     {"classmethod"}),
           rw.offset(mk("call", None, False), set()),
           rw.offset(mk("method", "alphalib.core.Base", True), set()),
           rw.offset(mk("method", "alphalib.core.Client", False),
                     {"staticmethod"})),
          (1, 1, 1, 0, 0, 0))


def case_plans():
    print("\n2. R1 ON THE CASES, THE OLD VERSION READ")
    api = oldapi.OldApi(("alphalib",), SITE)
    located = loc.locate(PROJECT, all_changes(), api)
    check("locate on the cases: 29 hits on 27 lines, one file; Child(...) "
          "meets Client.__init__'s rows through the inherited __init__; "
          "Client.parse and Client.from_url are hits; line 30 carries two; "
          "the call on a call's result (line 33) is not a hit, the inner "
          "call is; a method on what calling an instance returned (34) and "
          "on a name bound to it (36) are unverified",
          (len(located.hits), located.lines(), located.counts(),
           located.files_read,
           sorted(h.line for h in located.hits),
           sorted(h.line for h in located.hits
                  if h.confidence == "unverified")),
          (29, 27, {"resolved": 27, "referenced": 0, "unverified": 2}, 1,
           [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 27, 28, 29, 30, 30, 31, 32, 33, 34, 34, 35, 36], [34, 36]))
    plans = rw.r1_plans(located, all_changes(), api)
    diffs = rw.render(PROJECT, plans)
    check("every call's decision, names and text as written above",
          decisions(plans), EXPECTED)
    by_line = {p.hit.line: p for p in plans}
    check("15 rewrites, 9 withheld, 3 need nothing",
          sorted((a, sum(1 for p in plans if p.action == a))
                 for a in ("rewrite", "flag", "nothing")),
          [("flag", 9), ("nothing", 3), ("rewrite", 15)])
    check("the reasons say why: already by name, not passed, unpacking, "
          "overflow, positional-only, the function's flags, both ways",
          (by_line[11].reason, by_line[12].reason,
           "unpacks" in by_line[14].reason,
           "5 positional argument(s) and the old signature has 4 named"
           in by_line[16].reason and "already wrong" in by_line[16].reason,
           by_line[19].reason,
           "b was positional-only in the old version" in by_line[20].reason,
           by_line[21].reason,
           "went to *args" in by_line[27].reason
           and "rest's kind changed (variadic positional -> variadic "
               "keyword); not a case R1 covers" in by_line[27].reason,
           by_line[29].reason,
           by_line[23].reason, by_line[31].reason,
           "4 positional argument(s) and the old signature has 3 named"
           in by_line[32].reason and "already wrong" in by_line[32].reason),
          ("already passed by name: db, port",
           "the moved parameter(s) are not passed here: db, port",
           True, True,
           "the moved parameter(s) are not passed here: timeout",
           True,
           "the function also carries a flag: port became positional-only "
           "in 2.0.0; timeout was removed in 2.0.0",
           True,
           "port is passed by position and by keyword: the call is already "
           "wrong",
           "the function also carries a flag: timeout was added as required "
           "in 2.0.0",
           "retries is passed by position and by keyword: the call is "
           "already wrong", True))
    check("each plan names its function and carries its rows",
          (by_line[9].symbol, by_line[9].symbols, by_line[9].rows,
           by_line[17].symbol, by_line[17].rows, by_line[22].symbol,
           by_line[23].symbol, by_line[23].rows, by_line[28].symbol,
           by_line[21].rows),
          ("alphalib.core.Client.__init__", ("alphalib.core.Client.__init__",),
           (1, 2, 3), "alphalib.core.Base.send", (4,),
           "alphalib.core.Client.__init__", "alphalib.core.Client.from_url",
           (15, 21), "alphalib.core.Client.parse", (5, 19, 20)))
    check("one file's diff, fifteen source lines changed, the comment "
          "kept, nothing else touched, the two calls of line 30 both named",
          (len(diffs), diffs[0].path,
           sum(1 for ln in diffs[0].text.splitlines()
               if ln.startswith("-") and not ln.startswith("---")),
           sum(1 for ln in diffs[0].text.splitlines()
               if ln.startswith("+") and not ln.startswith("+++")),
           '+r = Client("h", port=1,  # the port' in diffs[0].text,
           "+           db=2)" in diffs[0].text,
           '-a = Client("localhost", 6380, 2)' in diffs[0].text,
           '+a = Client("localhost", port=6380, db=2)' in diffs[0].text,
           '+v = Client("h", port=1).send(b"t", timeout=5)' in diffs[0].text,
           '+y = Client("h", port=1)("x")' in diffs[0].text,
           '-p = Client.from_url("u", 5)' in diffs[0].text,
           '+z = Client("h", port=1)("x").send(b"t", 5)' in diffs[0].text,
           diffs[0].old == (PROJECT / CASES).read_text()),
          (1, CASES, 15, 15, True, True, True, True, True, True, False, True,
           True))
    check("the files were not written by render",
          'a = Client("localhost", 6380, 2)' in (PROJECT / CASES).read_text(),
          True)
    return plans, diffs


def case_without_old() -> None:
    print("\n3. THE SAME, THE OLD VERSION NOT READ")
    located = loc.locate(PROJECT, all_changes(), None)
    plans = rw.r1_plans(located, all_changes(), None,
                        "alphalib is not installed here")
    got = sorted((p.hit.line, p.action) for p in plans)
    check("Child(...) is no hit without the old version (the inherited "
          "__init__ is not known) and a.send(...) is unverified, so neither "
          "is planned; the Base.send explicit call is a plain call, still "
          "planned; every decision that needs the signature is withheld, "
          "the unpacking and the all-by-name case decided without it",
          got,
          [(9, "flag"), (10, "flag"), (11, "nothing"), (12, "flag"),
           (13, "flag"), (14, "flag"), (15, "flag"), (16, "flag"),
           (18, "flag"), (20, "flag"), (21, "flag"), (23, "flag"),
           (24, "flag"), (25, "flag"), (27, "flag"), (28, "flag"),
           (29, "flag"), (30, "flag"), (31, "flag"), (32, "flag"),
           (33, "flag"), (34, "flag"), (35, "flag")])
    reasons = {p.hit.line: p.reason for p in plans}
    check("the reason names what is missing",
          (reasons[9], reasons[14], reasons[11]),
          ("the old signature could not be read: alphalib is not installed "
           "here",
           "the call unpacks `*` or `**`: the positions are unknown",
           "already passed by name: db, port"))
    check("no rewrite without the signature: an empty diff",
          rw.render(PROJECT, plans), [])


def case_apply(tmp: pathlib.Path) -> None:
    print("\n4. APPLY, THEN RUN AGAIN: IDEMPOTENCE")
    copy = tmp / "rewrite_app"
    shutil.copytree(PROJECT, copy)
    api = oldapi.OldApi(("alphalib",), SITE)
    located = loc.locate(copy, all_changes(), api)
    plans = rw.r1_plans(located, all_changes(), api)
    diffs = rw.render(copy, plans)
    written = rw.apply(copy, diffs)
    text = (copy / CASES).read_text()
    check("apply writes the one file, and its text is the diff's new side",
          (written, text == diffs[0].new,
           'a = Client("localhost", port=6380, db=2)' in text,
           'Client("h", port=1,  # the port\n           db=2)' in text),
          ([CASES], True, True, True))
    located2 = loc.locate(copy, all_changes(), api)
    plans2 = rw.r1_plans(located2, all_changes(), api)
    check("the second run finds the same 29 hits, rewrites nothing, and "
          "the fifteen are now 'already passed by name' (sixteen with "
          "line 11); the nine withheld stay withheld",
          (len(located2.hits),
           sum(1 for p in plans2 if p.action == "rewrite"),
           sum(1 for p in plans2 if p.action == "nothing"
               and p.reason.startswith("already passed by name")),
           sum(1 for p in plans2 if p.action == "flag"),
           rw.render(copy, plans2)),
          (29, 0, 16, 9, []))
    check("the original fixture is untouched",
          'a = Client("localhost", 6380, 2)' in (PROJECT / CASES).read_text(),
          True)


def case_edges(tmp: pathlib.Path) -> None:
    print("\n4b. EDGES: NO FINAL NEWLINE, A BYTE-ORDER MARK, ONE CALL MEETING "
          "TWO DEFINING PATHS, THE LONG LIST")
    proj = tmp / "edges"
    proj.mkdir()
    (proj / "a.py").write_text('from alphalib import Client\n'
                               'c = Client("h", 1, 2)')      # no newline
    (proj / "b.py").write_bytes(("\ufefffrom alphalib import Client\n"
                                 "d = Client('h', 1)\n").encode("utf-8"))
    api = oldapi.OldApi(("alphalib",), SITE)
    rows = all_changes()
    located = loc.locate(proj, rows, api)
    plans = rw.r1_plans(located, rows, api)
    diffs = rw.render(proj, plans)
    text = "".join(d.text for d in diffs)
    check("a file whose last line has no newline: the diff ends its lines "
          "and marks the missing newline, so a.py's hunk does not run into "
          "b.py's header",
          (text.count("\\ No newline at end of file\n"),
           "-c = Client(\"h\", 1, 2)\n\\ No newline at end of file\n"
           "+c = Client(\"h\", port=1, db=2)\n\\ No newline at end of file\n"
           in text,
           "--- a/b.py" in text and text.index("--- a/b.py") > text.index(
               "+c = Client")),
          (2, True, True))
    (tmp / "edges.diff").write_text(text, encoding="utf-8")
    r = subprocess.run(["patch", "-p1", "--dry-run", "-i",
                        str(tmp / "edges.diff")], cwd=proj,
                       capture_output=True, text=True)
    check("patch -p1 accepts the diff for both files",
          (r.returncode, r.stdout.count("file a.py"),
           r.stdout.count("file b.py")), (0, 1, 1))
    check("the byte-order mark is not a change: b.py's diff touches the "
          "call's line alone",
          ([ln for ln in text.splitlines() if ln.startswith("-")
            and "b.py" not in ln and "a.py" not in ln],),
          (['-c = Client("h", 1, 2)', "-d = Client('h', 1)"],))
    rw.apply(proj, diffs)
    check("after apply the mark is still there and the call is named",
          (proj / "b.py").read_bytes()[:3] == b"\xef\xbb\xbf"
          and "d = Client('h', port=1)" in (proj / "b.py").read_text(
              encoding="utf-8-sig"), True)
    # one export path, two defining paths inside the range: the class moved
    # modules, and the new module's __init__ dropped db
    two = rows + [change(50, "alphalib.legacy.Client.__init__",
                         "PARAMETER_REMOVED", "db",
                         "alphalib.Client.__init__", score=0.5)]
    proj2 = tmp / "two"
    proj2.mkdir()
    (proj2 / "c.py").write_text('from alphalib import Client\n'
                                'e = Client("h", 1, 2)\n')
    located = loc.locate(proj2, two, api)
    plans = rw.r1_plans(located, two, api)
    check("a withheld rewrite writes no diff", rw.render(proj2, plans), [])
    check("one call, one decision from both functions' rows: withheld by the "
          "removal under the other path, the would-be from the signature "
          "the old version has, and both entries named",
          [(p.action, p.reason, p.would_be, p.symbol, p.symbols)
           for p in plans],
          [("flag", "the function also carries a flag: db was removed in "
                    "2.0.0", 'Client("h", port=1, db=2)',
            "alphalib.core.Client.__init__",
            ("alphalib.core.Client.__init__",
             "alphalib.legacy.Client.__init__"))])
    ranked = sorted(two, key=lambda c: (-c.score, c.symbol, c.kind,
                                        c.sub_target))
    report = rep.report("alphalib", "1.0.0", "2.0.0", [], ranked, located,
                        "d", "/p", [], plans)
    check("the decision prints under both functions' entries",
          report.count("-> withheld: the function also carries a flag: db "
                       "was removed in 2.0.0 | would be: "
                       'Client("h", port=1, db=2)'), 2)
    check("a long list of moved names is cut to six and a count",
          (rw._listed(set("abcdefgh")), rw._listed({"port", "db"})),
          ("a, b, c, d, e, f and 2 more", "db, port"))
    check("a multi-line call prints on one line in the report, its "
          "comment left out",
          rep.one_line('Client("h", port=1,  # the port\n           db=2)'),
          'Client("h", port=1, db=2)')


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def case_cli(tmp: pathlib.Path) -> None:
    print("\n5. THE COMMAND LINE: THE REPORT'S ACTIONS, --diff, --apply")
    bundle = tmp / "b.sqlite"
    write_bundle(bundle, all_changes())
    copy = tmp / "cli_app"
    shutil.copytree(PROJECT, copy)
    base = ["alphalib", str(copy), "--from", "1.0.0", "--to", "2.0.0",
            "--bundle", str(bundle), "--env", str(SITE)]
    code, out, err = run(base)
    lines = out.splitlines()
    check("exit 1; the report says what R1 did, in numbers",
          (code, "R1, pass by name: 15 call(s) rewritten, 9 withheld, "
                 "3 need nothing; the diff follows." in out), (1, True))
    # the decision prints under the FUNCTION's entry (Client.__init__),
    # not under the class's own row (the base removed), which the same
    # call also meets
    first = next(k for k, ln in enumerate(lines)
                 if "alphalib.core.Client.__init__   used by" in ln)
    i = next(k for k, ln in enumerate(lines)
             if k > first and ln.startswith("       cases.py:9:5 "))
    check("under a rewritten line, the call as it will read, under the "
          "function's entry",
          lines[i + 1].strip(),
          '-> rewritten: Client("localhost", port=6380, db=2)')
    j = next(k for k, ln in enumerate(lines)
             if k < first and ln.startswith("       cases.py:9:5 "))
    check("not under the class's entry for the removed base",
          lines[j + 1].startswith("       cases.py:"), True)
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       cases.py:21:5 "))
    check("under a withheld line, the reason and what it would be",
          lines[i + 1].strip(),
          "-> withheld: the function also carries a flag: port became "
          "positional-only in 2.0.0; timeout was removed in 2.0.0 | would "
          'be: connect("h", port=2, timeout=9)')
    i = next(k for k, ln in enumerate(lines)
             if k > first and ln.startswith("       cases.py:11:5 "))
    check("under a line that needs nothing, why",
          lines[i + 1].strip(),
          "-> needs nothing from R1: already passed by name: db, port")
    check("the diff is printed after the report, and the file is not "
          "written without --apply",
          ("--- a/cases.py" in out and "+++ b/cases.py" in out,
           '+a = Client("localhost", port=6380, db=2)' in out,
           'a = Client("localhost", 6380, 2)' in (copy / CASES).read_text()),
          (True, True, True))
    code, out, err = run(base + ["--diff", str(tmp / "r1.diff")])
    check("--diff writes the diff to the path and says so",
          (code, "--- a/cases.py" in out,
           f"diff written to {tmp / 'r1.diff'} (1 file(s))" in out,
           "+++ b/cases.py" in (tmp / "r1.diff").read_text()),
          (1, False, True, True))
    r = subprocess.run(["patch", "-p1", "--dry-run", "-i",
                        str(tmp / "r1.diff")], cwd=copy, capture_output=True,
                       text=True)
    check("patch -p1 accepts the written diff", r.returncode, 0)
    code, out, err = run(base + ["--apply"])
    check("--apply writes the file, prints the diff and says what it wrote",
          (code, "1 file(s) rewritten: cases.py" in out,
           "--- a/cases.py" in out,
           'a = Client("localhost", port=6380, db=2)'
           in (copy / CASES).read_text()), (1, True, True, True))
    code, out, err = run(base)
    check("the run after --apply: nothing to rewrite, 18 need nothing, "
          "no diff, still exit 1 for the flagged lines",
          (code, "R1, pass by name: 0 call(s) rewritten, 9 withheld, "
                 "18 need nothing." in out, "--- a/" in out),
          (1, True, False))
    code, out, err = run(base + ["--diff", str(tmp / "empty.diff")])
    check("--diff with nothing to rewrite writes an empty file and says so",
          (f"diff written to {tmp / 'empty.diff'} (0 file(s)); nothing to "
           "rewrite, so it is empty" in out,
           (tmp / "empty.diff").read_text()), (True, ""))
    other = tmp / "other-site"
    (other / "alphalib-0.8.0.dist-info").mkdir(parents=True)
    (other / "alphalib-0.8.0.dist-info" / "METADATA").write_text(
        "Name: alphalib\nVersion: 0.8.0\n")
    copy2 = tmp / "cli_app2"
    shutil.copytree(PROJECT, copy2)
    code, out, err = run(["alphalib", str(copy2), "--from", "1.0.0", "--to",
                          "2.0.0", "--bundle", str(bundle), "--env",
                          str(other)])
    check("without the old version: no rewrite, every candidate withheld "
          "with the reason, no diff",
          (code, "0 call(s) rewritten, 22 withheld, 1 need nothing" in out,
           "-> withheld: the old signature could not be read: the old "
           "version was not read (see the environment line above)" in out,
           "--- a/" in out), (1, True, True, False))


def main() -> None:
    case_parsing()
    case_plans()
    case_without_old()
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        case_apply(tmp)
        case_edges(tmp)
        case_cli(tmp)
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not ship R1 until this passes.")
        sys.exit(1)
    print("All checks passed. R1 names what the old signature names, and")
    print("withholds, with the reason, everywhere else.")


if __name__ == "__main__":
    main()
