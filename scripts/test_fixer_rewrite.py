"""
Do R1 and R2 rewrite exactly the calls §38.4 says, with the names and the
values the old signature and the rows give, and withhold with the reason
everywhere else?

    python scripts/test_fixer_rewrite.py

No database, no network, nothing imported from the fixtures. The project
is scripts/fixtures/rewrite_app/: cases.py, one call per line for R1
(and, since week 4, R2 on the same calls), and defaults.py, one call per
line for R2 (NOTES §38.15), against the pretend package alphalib 1.0.0
(scripts/fixtures/site/) with the forty-five changes of
scripts/test_fixer_locate.py (its twelve, the review's three, seven for
R1's cases, twenty-three for R2's and the review of R2). Client.__init__(self, host, port=6379,
db=0, retries=3) has port and db MOVED and retries re-defaulted 3 -> 5;
Base.send(self, payload, timeout=10) has timeout made keyword-only;
helper(a, b, /, c, *, d=1) has b MOVED, a positional-only parameter, and
d re-defaulted 1 -> 2; connect(host, port=1, timeout=5) has timeout
REMOVED and port both MOVED and made positional-only; spread(first,
*rest, flag=False) has first MOVED and rest's kind changed in a way R1
does not cover; the classmethod Client.from_url(cls, url, retries=3) has
retries MOVED and a timeout ADDED as required; the staticmethod
Client.parse(url, strict=False) has strict made keyword-only. In
alphalib.opts: fetch(url, timeout=None, retries=3, verify=True,
mode='fast', limit=DEFAULT_LIMIT, hooks=[], pool=None, tags=()) has every
default changed (timeout -> DEFAULT_TIMEOUT, retries -> 5, verify ->
False, mode -> "fast", limit -> 20, hooks -> None, pool -> Pool(), tags ->
None); shape(a, b=1, /, c=2, *, d=1e3) has b -> 2, d -> 1000.0 and c now
required; Session.__init__(self, host, port=6379, *, protocol=2,
keepalive=None) has protocol -> None and keepalive -> True; Session.get
(self, key, default=None) has default -> SENTINEL; ship(path,
compress=False) has compress -> True and a dest ADDED as required;
emit(items, sep=",") has items MOVED and a row on sep whose message
carries no values; wrap(x, y=2, /, z=3) has y, positional-only, made
required, z -> 4 and z MOVED.

Expected, written BEFORE the code ran. One decision per call, from both
rules; "would be" is the call as the diff will have it, and a withheld
rule's would-be its own edit (a rule keeps its edit when withheld only
under the function's flags, which withhold both rules).

cases.py (R1's lines; R2 meets Client.__init__'s retries and helper's d):

   9  Client("localhost", 6380, 2)      R1 port, db; R2 retries=3
        Client("localhost", port=6380, db=2, retries=3)
  10  Client("localhost", 6380)         R1 port; R2 retries=3
  11  Client(..., port=, db=)           R1 nothing (by name); R2 retries=3
  12  Client("localhost")               R1 nothing (not passed); R2 retries=3
  13  Client("h", 1, 2, 3)              R1 port, db, retries (R2's row sits
                                        in the run); R2 nothing: retries
                                        is passed by position
  14  Client(*d.args)                   both withheld: unpacking
  15  Client("h", 1, db=2)              R1 port; R2 retries=3
  16  Client("h", 1, 2, 3, 4)           both withheld: 5 positional, 4
                                        named, already wrong
  17  a.send(b"x", 5)                   R1 timeout, self offset; no R2 row
  18  Base.send(a, b"x", 5)             R1 timeout, no offset; no R2 row
  19  a.send(b"x")                      R1 nothing; no R2 row
  20  helper(1, 2, 3)                   R1 withheld, b positional-only;
                                        R2 d=1: helper(1, 2, 3, d=1)
  21  connect("h", 2, timeout=9)        R1 withheld: port positional-only,
                                        timeout removed; would be
                                        connect("h", port=2, timeout=9);
                                        no R2 row
  22  Child("h", 1, 2)                  Client's rows through the inherited
                                        __init__: R1 port, db; R2 retries=3
  23  Client.from_url("u", 5)           R1 withheld: timeout added as
                                        required; would be
                                        Client.from_url("u", retries=5),
                                        cls offset; no R2 row
  24  alphalib.Client("h", 1)           R1 port; R2 retries=3
  25  Client("h", 1,  # the port        R1 port, db; R2 retries=3 on the
             2)                         second line, comment kept
  27  spread("a", "b", "c", True)       R1 withheld: 4 positional, 1 named,
                                        the rest went to *args; rest's kind
                                        change is not R1's case. R2 (flag
                                        False -> True) withheld by that
                                        same flag, not by the overflow,
                                        which *rest takes: would be
                                        spread("a", "b", "c", True,
                                        flag=False)
  28  Client.parse("u", True)           R1 strict, no offset (staticmethod)
  29  Client("h", 1, port=2)            both withheld: port by position and
                                        keyword, already wrong
  30  Client("h", 1).send(b"t", 5)      two calls at one column: the
                                        constructor R1 port + R2 retries=3,
                                        the method R1 timeout:
                                        Client("h", port=1, retries=3)
                                        .send(b"t", timeout=5)
  31  Client("h", 1, 2, 3, retries=9)   R1 withheld: retries by position
                                        and keyword; R2 nothing: retries
                                        passed (by name)
  32  helper(1, 2, 3, 4)                both withheld: 4 positional, 3
                                        named (d keyword-only), wrong
  33  Client("h", 1)("x")               the inner call: R1 port, R2 retries
  34  Client("h", 1)("x").send(...)     the constructor only; the .send is
                                        unverified
  35  r = Client("h", 1)("x")           the constructor; r is unknown
  36  r.send(b"t", 6)                   unverified, no decision

  R1: 15 rewritten, 9 withheld, 3 need nothing, as in week 3. R2 on the
  20 calls it meets: 13 pinned, 5 withheld, 2 need nothing. The file's
  calls: 18 rewritten, 8 withheld (31 counts as withheld: R1 is), 1
  needs nothing; 18 source lines change (25 and 26 for one call).

defaults.py (R2's lines, §38.15's table; no R1 row on these functions):

   7  fetch("u")                        pinned timeout=None, retries=3,
                                        verify=True, pool=None, tags=();
                                        not pinned: limit (DEFAULT_LIMIT is
                                        not a plain value), hooks ([] is a
                                        mutable container); mode
                                        changed only in spelling
   8  fetch("u", 5, 9)                  verify=True, pool=None, tags=():
                                        timeout, retries by position
   9  fetch("u", retries=1, verify=False, pool=None, timeout=1, tags=(1,))
                                        left to you: nothing to pin, limit
                                        and hooks listed with their values
  10  fetch("u", **kw)                  withheld: unpacking
  11  fetch(*args)                      withheld: unpacking
  12  shape(1)                          c=2, now required, the old default
                                        from the signature; b not pinned,
                                        positional-only; d only in spelling
  13  shape(1, 2, 3)                    nothing: b, c by position; d only in
                                        spelling
  14  Session("h")                      protocol=2, keepalive=None
                                        (keyword-only, signature order)
  15  Session("h", 1, protocol=3)       keepalive=None
  16  h.get("k")                        default=None, self offset
  17  Session.get(h, "k")               default=None, no offset
  18  ship("p")                         withheld: dest added as required;
                                        would be ship("p", compress=False)
  19  alphalib.opts.fetch("u", timeout=2)
                                        retries=3, verify=True, pool=None,
                                        tags=()
  20  fetch("u", 1, 2, ..., 9)          withheld: 10 positional, 9 named,
                                        already wrong
  21  fetch(                            one pin per line, the trailing
        "u",                            comma last:
        retries=1,                        "u", / retries=1, / timeout=None,
      )                                   / verify=True, / pool=None, /
                                          tags=(), / )
  25  fetch(                            the same, and the comment on the
        "u",  # the url                 first line is not copied to the
        retries=1,                      new ones
      )
  29  fetch(                            the comment on the LAST argument's
        "u",                            comma stays on that argument; the
        retries=1,  # retry once        pins follow it, one per line
      )
  33  wrap(1)                           withheld: y, positional-only, is
                                        now required and no pin can answer
                                        it, so the call will not run on
                                        2.0.0 whatever else is done; would
                                        be wrap(1, z=3); R1 (z moved, not
                                        passed) needs nothing and carries
                                        the same flag
  34  emit(x for x in y)                R1 alone names the generator, which
                                        takes its parentheses:
                                        emit(items=(x for x in y)); R2 left
                                        to you: sep's row has no values
  35  fetch()                           every pin, into an empty call
  36  fetch(                            the blank line between arguments is
        "u",                            not copied before each pin
        (blank)
        retries=1,
      )
  16, 17 also meet a row on `ghost`, a name Session.get does not have in
  the old version the tool reads: not pinned, said so.

  13 pinned, 6 withheld, 2 left to you, 1 needs nothing, twenty-two
  calls. The project: R1 16/9/4, R2 26 pinned, 10 withheld, 2 left to
  you, 3 need nothing. Without the old version: R1 24 withheld, 1
  nothing; R2 0 pinned, 37 withheld, 2 left to you, 1 need nothing (line 31's retries=9
  by name), the unpacking and the all-named cases decided without the
  signature and everything else "the old signature could not be read";
  h.get("k") still resolves without the version (Session is imported by
  its defining path and get is defined on it), where a.send needs the
  MRO. Idempotence: both diffs applied to a copy, locate and the rules
  run again: no rewrite, R1 0/9/20, R2 0 pinned, 10 withheld, 13 left to
  you (the calls whose unpinnable rows remain), 18 need nothing, every
  former pin "already passed", an empty diff. The diff
  applies with `patch -p1`, also for a file whose last line has no
  newline; a byte-order mark survives `--apply`; one call meeting two
  defining paths gets one decision from both their rows, from both rules.
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
    SITE, change, changes, extra_changes, r2_changes, rewrite_changes,
    write_bundle)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE / "fixtures" / "rewrite_app"
CASES, DEFAULTS = "cases.py", "defaults.py"

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


def all_changes():
    return changes() + extra_changes() + rewrite_changes() + r2_changes()


def decisions(plans):
    """(file, line, R1's action, names, R2's action, pins, the call as
    the diff will have it) per plan, in file and line order."""
    return sorted((p.hit.file, p.hit.line, p.r1.action, p.names,
                   p.r2.action, p.pins, p.would_be) for p in plans)


def counts(plans, rule):
    """(rewritten, withheld, left to you, nothing) for one rule."""
    return tuple(sum(1 for p in plans if getattr(p, rule).action == a)
                 for a in ("rewrite", "flag", "left", "nothing"))


C, D = CASES, DEFAULTS
EXPECTED = [
    (C, 9, "rewrite", ("port", "db"), "rewrite", ("retries=3",),
     'Client("localhost", port=6380, db=2, retries=3)'),
    (C, 10, "rewrite", ("port",), "rewrite", ("retries=3",),
     'Client("localhost", port=6380, retries=3)'),
    (C, 11, "nothing", (), "rewrite", ("retries=3",),
     'Client("localhost", port=6380, db=2, retries=3)'),
    (C, 12, "nothing", (), "rewrite", ("retries=3",),
     'Client("localhost", retries=3)'),
    (C, 13, "rewrite", ("port", "db", "retries"), "nothing", (),
     'Client("h", port=1, db=2, retries=3)'),
    (C, 14, "flag", (), "flag", (), None),
    (C, 15, "rewrite", ("port",), "rewrite", ("retries=3",),
     'Client("h", port=1, db=2, retries=3)'),
    (C, 16, "flag", (), "flag", (), None),
    (C, 17, "rewrite", ("timeout",), "", (), 'a.send(b"x", timeout=5)'),
    (C, 18, "rewrite", ("timeout",), "", (),
     'Base.send(a, b"x", timeout=5)'),
    (C, 19, "nothing", (), "", (), None),
    (C, 20, "flag", (), "rewrite", ("d=1",), 'helper(1, 2, 3, d=1)'),
    (C, 21, "flag", ("port",), "", (), None),
    (C, 22, "rewrite", ("port", "db"), "rewrite", ("retries=3",),
     'Child("h", port=1, db=2, retries=3)'),
    (C, 23, "flag", ("retries",), "", (), None),
    (C, 24, "rewrite", ("port",), "rewrite", ("retries=3",),
     'alphalib.Client("h", port=1, retries=3)'),
    (C, 25, "rewrite", ("port", "db"), "rewrite", ("retries=3",),
     'Client("h", port=1,  # the port\n           db=2, retries=3)'),
    (C, 27, "flag", (), "flag", ("flag=False",), None),
    (C, 28, "rewrite", ("strict",), "", (), 'Client.parse("u", strict=True)'),
    (C, 29, "flag", (), "flag", (), None),
    (C, 30, "rewrite", ("port",), "rewrite", ("retries=3",),
     'Client("h", port=1, retries=3)'),
    (C, 30, "rewrite", ("timeout",), "", (),
     'Client("h", port=1, retries=3).send(b"t", timeout=5)'),
    (C, 31, "flag", (), "nothing", (), None),
    (C, 32, "flag", (), "flag", (), None),
    (C, 33, "rewrite", ("port",), "rewrite", ("retries=3",),
     'Client("h", port=1, retries=3)'),
    (C, 34, "rewrite", ("port",), "rewrite", ("retries=3",),
     'Client("h", port=1, retries=3)'),
    (C, 35, "rewrite", ("port",), "rewrite", ("retries=3",),
     'Client("h", port=1, retries=3)'),
    (D, 7, "", (), "rewrite",
     ("timeout=None", "retries=3", "verify=True", "pool=None", "tags=()"),
     'fetch("u", timeout=None, retries=3, verify=True, pool=None, tags=())'),
    (D, 8, "", (), "rewrite", ("verify=True", "pool=None", "tags=()"),
     'fetch("u", 5, 9, verify=True, pool=None, tags=())'),
    (D, 9, "", (), "left", (), None),
    (D, 10, "", (), "flag", (), None),
    (D, 11, "", (), "flag", (), None),
    (D, 12, "", (), "rewrite", ("c=2",), 'shape(1, c=2)'),
    (D, 13, "", (), "nothing", (), None),
    (D, 14, "", (), "rewrite", ("protocol=2", "keepalive=None"),
     'Session("h", protocol=2, keepalive=None)'),
    (D, 15, "", (), "rewrite", ("keepalive=None",),
     'Session("h", 1, protocol=3, keepalive=None)'),
    (D, 16, "", (), "rewrite", ("default=None",), 'h.get("k", default=None)'),
    (D, 17, "", (), "rewrite", ("default=None",),
     'Session.get(h, "k", default=None)'),
    (D, 18, "", (), "flag", ("compress=False",), None),
    (D, 19, "", (), "rewrite",
     ("retries=3", "verify=True", "pool=None", "tags=()"),
     'alphalib.opts.fetch("u", timeout=2, retries=3, verify=True, '
     'pool=None, tags=())'),
    (D, 20, "", (), "flag", (), None),
    (D, 21, "", (), "rewrite",
     ("timeout=None", "verify=True", "pool=None", "tags=()"),
     'fetch(\n    "u",\n    retries=1,\n    timeout=None,\n    verify=True,'
     '\n    pool=None,\n    tags=(),\n)'),
    (D, 25, "", (), "rewrite",
     ("timeout=None", "verify=True", "pool=None", "tags=()"),
     'fetch(\n    "u",  # the url\n    retries=1,\n    timeout=None,\n'
     '    verify=True,\n    pool=None,\n    tags=(),\n)'),
    (D, 29, "", (), "rewrite",
     ("timeout=None", "verify=True", "pool=None", "tags=()"),
     'fetch(\n    "u",\n    retries=1,  # retry once\n    timeout=None,\n'
     '    verify=True,\n    pool=None,\n    tags=(),\n)'),
    (D, 33, "nothing", (), "flag", ("z=3",), None),
    (D, 34, "rewrite", ("items",), "left", (), 'emit(items=(x for x in y))'),
    (D, 35, "", (), "rewrite",
     ("timeout=None", "retries=3", "verify=True", "pool=None", "tags=()"),
     'fetch(timeout=None, retries=3, verify=True, pool=None, tags=())'),
    (D, 36, "", (), "rewrite",
     ("timeout=None", "verify=True", "pool=None", "tags=()"),
     'fetch(\n    "u",\n\n    retries=1,\n    timeout=None,\n'
     '    verify=True,\n    pool=None,\n    tags=(),\n)'),
]


def case_parsing() -> None:
    print("\n1. GRIFFE'S WORDS FOR A KIND CHANGE, THE SELF OFFSET, A PLAIN "
          "LITERAL")
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
    check("a plain literal: numbers, strings, None, booleans, tuples; not "
          "a name, a call, an empty list or dict, a set() or an expression",
          tuple(rw.plain(v) for v in
                ("2", "None", "True", "'fast'", "1e3", "-1", "()",
                 "(1, 2)", "DEFAULT_LIMIT", "Pool()", "[]", "{}", "set()",
                 "1 + 1", "[1]", "{1: 2}", "([1],)", "(1, (2, []))",
                 "((1,),)")),
          (True, True, True, True, True, True, True, True, False, False,
           False, False, False, False, False, False, False, False, True))
    check("why a value is not pinned, in words",
          (rw.why_not_plain("DEFAULT_LIMIT"), rw.why_not_plain("[]"),
           rw.why_not_plain("{}"), rw.why_not_plain("Pool()")),
          ("is not a plain value", "is a mutable container",
           "is a mutable container", "is not a plain value"))
    check("a changed default that is cosmetic: the same value and type "
          "spelled differently; 1 -> 1.0 and 1 -> True are not cosmetic",
          tuple(rw.cosmetic(a, b) for a, b in
                (("'fast'", '"fast"'), ("1e3", "1000.0"), ("1", "1.0"),
                 ("1", "True"), ("None", "DEFAULT"), ("2", "None"),
                 ("()", "None"))),
          (True, True, False, False, False, False, False))
    check("a call spanning lines prints on one line, its comments out and "
          "its line breaks folded",
          (rep.one_line('fetch(\n    "u",  # the url\n    tags=(),\n)'),
           rep.one_line('Client("h", port=1,  # the port\n           db=2)')),
          ('fetch("u", tags=())', 'Client("h", port=1, db=2)'))


def case_plans():
    print("\n2. R1 AND R2 ON THE CASES, THE OLD VERSION READ")
    api = oldapi.OldApi(("alphalib",), SITE)
    located = loc.locate(PROJECT, all_changes(), api)
    check("locate on the project: 50 hits on 48 lines in two files; "
          "cases.py as in week 3 (29 on 27, two unverified), defaults.py 21 "
          "on 21, h.get resolved through the Session instance",
          (len(located.hits), located.lines(), located.counts(),
           located.files_read,
           sorted(h.line for h in located.hits if h.file == DEFAULTS),
           sorted(h.line for h in located.hits
                  if h.confidence == "unverified")),
          (50, 48, {"resolved": 48, "referenced": 0, "unverified": 2}, 2,
           [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25,
            29, 33, 34, 35, 36],
           [34, 36]))
    plans = rw.decide(located, all_changes(), api)
    diffs = rw.render(PROJECT, plans)
    check("every call's decision, names, pins and text as written above",
          decisions(plans), EXPECTED)
    check("R1: 16 rewritten, 9 withheld, 4 need nothing; R2: 26 pinned, 10 "
          "withheld, 2 left to you, 3 need nothing; the calls: 32 "
          "rewritten, 13 withheld, 1 left to you, 2 need nothing",
          (counts(plans, "r1"), counts(plans, "r2"),
           tuple(sum(1 for p in plans if p.action == a)
                 for a in ("rewrite", "flag", "left", "nothing"))),
          ((16, 9, 0, 4), (26, 10, 2, 3), (32, 13, 1, 2)))
    by_line = {(p.hit.file, p.hit.line): p for p in plans}
    c, d = (lambda n: by_line[(CASES, n)]), (lambda n: by_line[(DEFAULTS, n)])
    check("R1's reasons say why, as in week 3: already by name, not "
          "passed, unpacking, overflow, positional-only, the function's "
          "flags, both ways",
          (c(11).r1.reason, c(12).r1.reason,
           "unpacks" in c(14).r1.reason,
           "5 positional argument(s) and the old signature has 4 named"
           in c(16).r1.reason and "already wrong" in c(16).r1.reason,
           c(19).r1.reason,
           "b was positional-only in the old version" in c(20).r1.reason,
           c(21).r1.reason,
           "went to *args" in c(27).r1.reason
           and "rest's kind changed (variadic positional -> variadic "
               "keyword); not a case R1 covers" in c(27).r1.reason,
           c(29).r1.reason, c(23).r1.reason, c(31).r1.reason,
           "4 positional argument(s) and the old signature has 3 named"
           in c(32).r1.reason and "already wrong" in c(32).r1.reason),
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
    check("R2's reasons say why: the pins with their new values, what is "
          "not pinned and why, what changed only in spelling, already "
          "passed, unpacking, already wrong, the function's flag",
          (c(9).r2.reason, c(13).r2.reason, c(14).r2.reason,
           c(16).r2.reason, c(29).r2.reason, c(31).r2.reason,
           d(7).r2.reason, d(9).r2.reason, d(12).r2.reason,
           d(13).r2.reason, d(18).r2.reason, d(20).r2.reason,
           d(16).r2.reason, d(33).r2.reason, d(34).r1.reason,
           d(34).r2.reason, d(33).r1.reason),
          ("retries=3 (now 5)",
           "already passed: retries",
           "the call unpacks `*` or `**`: it may already pass retries",
           "the call passes 5 positional argument(s) and the old signature "
           "has 4 named one(s): the call was already wrong",
           "port is passed by position and by keyword: the call is already "
           "wrong",
           "already passed: retries",
           "timeout=None (now DEFAULT_TIMEOUT), retries=3 (now 5), "
           "verify=True (now False), pool=None (now Pool()), tags=() (now "
           "None); not pinned: limit (old default DEFAULT_LIMIT is not a "
           "plain value; now 20), hooks (old default [] is a mutable "
           "container; now None); changed only in spelling: mode ('fast' -> "
           '"fast")',
           "limit (old default DEFAULT_LIMIT is not a plain value; now 20), "
           "hooks (old default [] is a mutable container; now None); "
           "already passed: pool, retries, tags, timeout, verify; changed "
           "only in spelling: mode ('fast' -> \"fast\")",
           "c=2 (now required); not pinned: b (old default 1 is "
           "positional-only in the old version: no name to pass; now 2); "
           "changed only in spelling: d (1e3 -> 1000.0)",
           "already passed: b, c; changed only in spelling: d (1e3 -> "
           "1000.0)",
           "the function also carries a flag: dest was added as required "
           "in 2.0.0",
           "the call passes 10 positional argument(s) and the old signature "
           "has 9 named one(s): the call was already wrong",
           "default=None (now SENTINEL); not pinned: ghost (old default 0 "
           "is not a parameter of the old signature the tool read; now 1)",
           "the function also carries a flag: y is now required in 2.0.0 "
           "and cannot be pinned (old default 2 is positional-only in the "
           "old version: no name to pass)",
           "items",
           "sep (old and new value could not be read from griffe's "
           "message)",
           "the moved parameter(s) are not passed here: z; flagged: y is "
           "now required in 2.0.0 and cannot be pinned (old default 2 is "
           "positional-only in the old version: no name to pass)"))
    check("a withheld rule still says what it would be, its own edit: R1 "
          "on 21 and 23, R2 under ship's flag and spread's; none where the "
          "reason stopped the computation (20, 14, 32)",
          (c(20).r1.would_be, c(21).r1.would_be, c(23).r1.would_be,
           d(18).r2.would_be, c(14).r1.would_be, c(14).r2.would_be,
           c(32).r2.would_be, c(27).r2.would_be, d(33).r2.would_be,
           c(27).r2.reason),
          (None, 'connect("h", port=2, timeout=9)',
           'Client.from_url("u", retries=5)', 'ship("p", compress=False)',
           None, None, None, 'spread("a", "b", "c", True, flag=False)',
           'wrap(1, z=3)',
           "the function also carries a flag: rest's kind changed (variadic "
           "positional -> variadic keyword); not a case R1 covers"))
    check("each plan names its function and carries its rows",
          (c(9).symbol, c(9).symbols, c(9).rows, c(17).symbol, c(17).rows,
           c(22).symbol, c(23).symbol, c(23).rows, c(28).symbol, c(21).rows,
           d(7).symbol, d(7).rows, d(16).symbol, d(16).rows),
          ("alphalib.core.Client.__init__", ("alphalib.core.Client.__init__",),
           (1, 2, 3), "alphalib.core.Base.send", (4,),
           "alphalib.core.Client.__init__", "alphalib.core.Client.from_url",
           (15, 21), "alphalib.core.Client.parse", (5, 19, 20),
           "alphalib.opts.fetch", (23, 24, 25, 26, 27, 28, 29, 30),
           "alphalib.opts.Session.get", (36, 40)))
    by_path = {df.path: df for df in diffs}
    ct, dt = by_path[CASES].text, by_path[DEFAULTS].text
    minus = lambda t: sum(1 for ln in t.splitlines()  # noqa: E731
                          if ln.startswith("-") and not ln.startswith("---"))
    plus = lambda t: sum(1 for ln in t.splitlines()  # noqa: E731
                         if ln.startswith("+") and not ln.startswith("+++"))
    check("two files' diffs: cases.py changes 18 source lines, the comment "
          "kept and the pin after it, nothing else touched, the two calls "
          "of line 30 both edited; defaults.py changes 9 calls, the "
          "multi-line ones pinned a line at a time, the comment not copied",
          (sorted(by_path), minus(ct), plus(ct),
           '+r = Client("h", port=1,  # the port' in ct,
           "+           db=2, retries=3)" in ct,
           '-a = Client("localhost", 6380, 2)' in ct,
           '+a = Client("localhost", port=6380, db=2, retries=3)' in ct,
           '+v = Client("h", port=1, retries=3).send(b"t", timeout=5)' in ct,
           '+y = Client("h", port=1, retries=3)("x")' in ct,
           '-p = Client.from_url("u", 5)' in ct,
           '+z = Client("h", port=1, retries=3)("x").send(b"t", 5)' in ct,
           "+m = helper(1, 2, 3, d=1)" in ct,
           by_path[CASES].old == (PROJECT / CASES).read_text(),
           minus(dt), plus(dt),
           dt.count("+    timeout=None,\n+    verify=True,\n+    pool=None,\n"
                    "+    tags=(),\n )"),
           '+f = shape(1, c=2)' in dt,
           '+k = Session.get(h, "k", default=None)' in dt,
           dt.count("# the url"), dt.count("# retry once"),
           "+t = emit(items=(x for x in y))" in dt,
           "+u = fetch(timeout=None, retries=3, verify=True, pool=None, "
           "tags=())" in dt),
          ([CASES, DEFAULTS], 18, 18, True, True, True, True, True, True,
           False, True, True, True, 10, 26, 4, True, True, 1, 1, True,
           True))
    check("the files were not written by render",
          ('a = Client("localhost", 6380, 2)' in (PROJECT / CASES).read_text(),
           'a = fetch("u")\n' in (PROJECT / DEFAULTS).read_text()),
          (True, True))
    compile(by_path[DEFAULTS].new, DEFAULTS, "exec")
    compile(by_path[CASES].new, CASES, "exec")
    return plans, diffs


def case_without_old() -> None:
    print("\n3. THE SAME, THE OLD VERSION NOT READ")
    located = loc.locate(PROJECT, all_changes(), None)
    plans = rw.decide(located, all_changes(), None,
                      "alphalib is not installed here")
    got = sorted((p.hit.file, p.hit.line, p.r1.action, p.r2.action)
                 for p in plans)
    check("Child(...) is no hit without the old version and a.send(...) "
          "and line 30's .send are unverified, so none is planned; h.get "
          "still resolves (Session is imported by its defining path and get "
          "is defined on it); every decision that needs the signature is "
          "withheld; R1 decides the unpacking and the all-by-name case "
          "without it, R2 the unpacking and the all-named cases",
          got,
          [(C, 9, "flag", "flag"), (C, 10, "flag", "flag"),
           (C, 11, "nothing", "flag"), (C, 12, "flag", "flag"),
           (C, 13, "flag", "flag"), (C, 14, "flag", "flag"),
           (C, 15, "flag", "flag"), (C, 16, "flag", "flag"),
           (C, 18, "flag", ""), (C, 20, "flag", "flag"), (C, 21, "flag", ""),
           (C, 23, "flag", ""), (C, 24, "flag", "flag"),
           (C, 25, "flag", "flag"), (C, 27, "flag", "flag"),
           (C, 28, "flag", ""),
           (C, 29, "flag", "flag"), (C, 30, "flag", "flag"),
           (C, 31, "flag", "nothing"),
           (C, 32, "flag", "flag"), (C, 33, "flag", "flag"),
           (C, 34, "flag", "flag"), (C, 35, "flag", "flag"),
           (D, 7, "", "flag"), (D, 8, "", "flag"), (D, 9, "", "left"),
           (D, 10, "", "flag"), (D, 11, "", "flag"), (D, 12, "", "flag"),
           (D, 13, "", "flag"), (D, 14, "", "flag"), (D, 15, "", "flag"),
           (D, 16, "", "flag"), (D, 17, "", "flag"), (D, 18, "", "flag"),
           (D, 19, "", "flag"),
           (D, 20, "", "flag"), (D, 21, "", "flag"), (D, 25, "", "flag"),
           (D, 29, "", "flag"), (D, 33, "flag", "flag"),
           (D, 34, "flag", "left"),
           (D, 35, "", "flag"), (D, 36, "", "flag")])
    check("R1 24 withheld, 1 need nothing; R2 0 pinned, 37 withheld, 2 "
          "left to you, 1 need nothing",
          (counts(plans, "r1"), counts(plans, "r2")),
          ((0, 24, 0, 1), (0, 37, 2, 1)))
    reasons = {(p.hit.file, p.hit.line): p for p in plans}
    check("the reason names what is missing",
          (reasons[(C, 9)].r1.reason, reasons[(C, 14)].r1.reason,
           reasons[(C, 11)].r1.reason, reasons[(C, 11)].r2.reason,
           reasons[(C, 31)].r2.reason, reasons[(D, 9)].r2.reason,
           reasons[(D, 10)].r2.reason),
          ("the old signature could not be read: alphalib is not installed "
           "here",
           "the call unpacks `*` or `**`: the positions are unknown",
           "already passed by name: db, port",
           "the old signature could not be read: alphalib is not installed "
           "here",
           "already passed: retries",
           "limit (old default DEFAULT_LIMIT is not a plain value; now 20), "
           "hooks (old default [] is a mutable container; now None); "
           "already passed: pool, retries, tags, timeout, verify; changed "
           "only in spelling: mode ('fast' -> \"fast\")",
           "the call unpacks `*` or `**`: it may already pass pool, "
           "retries, tags, timeout, verify"))
    check("no rewrite without the signature: an empty diff",
          rw.render(PROJECT, plans), [])


def case_apply(tmp: pathlib.Path) -> None:
    print("\n4. APPLY, THEN RUN AGAIN: IDEMPOTENCE")
    copy = tmp / "rewrite_app"
    shutil.copytree(PROJECT, copy)
    api = oldapi.OldApi(("alphalib",), SITE)
    located = loc.locate(copy, all_changes(), api)
    plans = rw.decide(located, all_changes(), api)
    diffs = rw.render(copy, plans)
    written = rw.apply(copy, diffs)
    text = (copy / CASES).read_text()
    dtext = (copy / DEFAULTS).read_text()
    check("apply writes the two files, and their text is the diffs' new "
          "side",
          (written, text == diffs[0].new, dtext == diffs[1].new,
           'a = Client("localhost", port=6380, db=2, retries=3)' in text,
           'Client("h", port=1,  # the port\n           db=2, retries=3)'
           in text,
           'h = Session("h", protocol=2, keepalive=None)' in dtext,
           'retries=1,  # retry once\n    timeout=None,' in dtext),
          ([CASES, DEFAULTS], True, True, True, True, True, True))
    located2 = loc.locate(copy, all_changes(), api)
    plans2 = rw.decide(located2, all_changes(), api)
    check("the second run finds the same 50 hits, rewrites nothing: R1 0 "
          "rewritten, 9 withheld, 19 need nothing; R2 0 pinned, 10 "
          "withheld, 13 left to you (the calls whose unpinnable rows "
          "remain: limit and hooks on nine fetch calls, b on shape(1, c=2), "
          "ghost on the two get calls, sep on emit), 18 need nothing, every "
          "former pin 'already passed'",
          (len(located2.hits), counts(plans2, "r1"), counts(plans2, "r2"),
           sum(1 for p in plans2 if p.r2.action == "nothing"
               and p.r2.reason.startswith("already passed")),
           sum(1 for p in plans2 if p.r1.action == "nothing"
               and p.r1.reason.startswith("already passed by name")),
           rw.render(copy, plans2)),
          (50, (0, 9, 0, 20), (0, 10, 13, 18), 18, 17, []))
    check("the original fixture is untouched",
          ('a = Client("localhost", 6380, 2)' in (PROJECT / CASES).read_text(),
           'a = fetch("u")\n' in (PROJECT / DEFAULTS).read_text()),
          (True, True))


def case_edges(tmp: pathlib.Path) -> None:
    print("\n4b. EDGES: NO FINAL NEWLINE, A BYTE-ORDER MARK, ONE CALL MEETING "
          "TWO DEFINING PATHS, THE LONG LIST, A LONG VALUE")
    proj = tmp / "edges"
    proj.mkdir()
    (proj / "a.py").write_text('from alphalib import Client\n'
                               'c = Client("h", 1, 2)')      # no newline
    (proj / "b.py").write_bytes(("﻿from alphalib import Client\n"
                                 "d = Client('h', 1)\n").encode("utf-8"))
    api = oldapi.OldApi(("alphalib",), SITE)
    rows = all_changes()
    located = loc.locate(proj, rows, api)
    plans = rw.decide(located, rows, api)
    diffs = rw.render(proj, plans)
    text = "".join(d.text for d in diffs)
    check("a file whose last line has no newline: the diff ends its lines "
          "and marks the missing newline, so a.py's hunk does not run into "
          "b.py's header",
          (text.count("\\ No newline at end of file\n"),
           "-c = Client(\"h\", 1, 2)\n\\ No newline at end of file\n"
           "+c = Client(\"h\", port=1, db=2, retries=3)\n"
           "\\ No newline at end of file\n" in text,
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
    check("after apply the mark is still there and the call is named and "
          "pinned",
          (proj / "b.py").read_bytes()[:3] == b"\xef\xbb\xbf"
          and "d = Client('h', port=1, retries=3)" in (proj / "b.py").read_text(
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
    plans = rw.decide(located, two, api)
    check("a withheld rewrite writes no diff", rw.render(proj2, plans), [])
    check("one call, one decision from both functions' rows, from both "
          "rules: withheld by the removal under the other path, each "
          "rule's would-be its own edit (neither applies) from the "
          "signature the old version has, and both entries named",
          [(p.action, p.r1.action, p.r1.reason, p.r1.would_be, p.r2.action,
            p.r2.reason, p.r2.would_be, p.symbol, p.symbols)
           for p in plans],
          [("flag", "flag", "the function also carries a flag: db was "
            "removed in 2.0.0", 'Client("h", port=1, db=2)',
            "flag", "the function also carries a flag: db was removed in "
            "2.0.0", 'Client("h", 1, 2, retries=3)',
            "alphalib.core.Client.__init__",
            ("alphalib.core.Client.__init__",
             "alphalib.legacy.Client.__init__"))])
    ranked = sorted(two, key=lambda c: (-c.score, c.symbol, c.kind,
                                        c.sub_target))
    report = rep.report("alphalib", "1.0.0", "2.0.0", [], ranked, located,
                        "d", "/p", [], plans)
    check("the decision prints under both functions' entries, three lines "
          "each",
          (report.count("-> withheld"),
           report.count("R1 withheld: the function also carries a flag: db "
                        "was removed in 2.0.0 | would be: "
                        'Client("h", port=1, db=2)'),
           report.count("R2 withheld: the function also carries a flag: db "
                        "was removed in 2.0.0 | would be: "
                        'Client("h", 1, 2, retries=3)')),
          (2, 2, 2))
    check("a long list of moved names is cut to six and a count",
          (rw._listed(set("abcdefgh")), rw._listed({"port", "db"})),
          ("a, b, c, d, e, f and 2 more", "db, port"))
    check("a multi-line call prints on one line in the report, its "
          "comment left out",
          rep.one_line('Client("h", port=1,  # the port\n           db=2)'),
          'Client("h", port=1, db=2)')
    # one name resolving to two functions the old version HAS, with
    # different signatures (a compatibility shim): the edit computed on
    # one may be wrong for the other, so both rules are withheld
    proj5 = tmp / "shim"
    proj5.mkdir()
    (proj5 / "f.py").write_text('try:\n    from alphalib import Client\n'
                                'except ImportError:\n'
                                '    from alphalib.opts import Session as '
                                'Client\n'
                                'c = Client("h", 1)\n')
    located5 = loc.locate(proj5, rows, api)
    plans5 = rw.decide(located5, rows, api)
    diffs5 = rw.render(proj5, plans5)
    check("two old signatures for one name: both rules withheld with the "
          "reason, each would-be from the first signature, both functions "
          "named, no diff",
          ([(p.action, p.r1.action, p.r1.reason, p.r1.would_be, p.r2.action,
             p.r2.reason, p.r2.would_be, p.symbols) for p in plans5],
           diffs5),
          ([("flag", "flag",
             "the function also carries a flag: the call resolves to "
             "alphalib.core.Client.__init__ and "
             "alphalib.opts.Session.__init__, whose old signatures differ",
             'Client("h", port=1)', "flag",
             "the function also carries a flag: the call resolves to "
             "alphalib.core.Client.__init__ and "
             "alphalib.opts.Session.__init__, whose old signatures differ",
             'Client("h", 1, retries=3)',
             ("alphalib.core.Client.__init__",
              "alphalib.opts.Session.__init__"))], []))
    # a form feed (or any character str.splitlines breaks at and Python
    # does not) must not put the no-newline marker inside a hunk
    proj6 = tmp / "formfeed"
    proj6.mkdir()
    (proj6 / "g.py").write_text('from alphalib.opts import fetch\n\x0c\n'
                                'x = 1\ns = "a\x1cb"\na = fetch("u")\n')
    located6 = loc.locate(proj6, rows, api)
    diffs6 = rw.render(proj6, rw.decide(located6, rows, api))
    (tmp / "ff.diff").write_text(diffs6[0].text, encoding="utf-8")
    r = subprocess.run(["patch", "-p1", "--dry-run", "-i",
                        str(tmp / "ff.diff")], cwd=proj6,
                       capture_output=True, text=True)
    check("a form feed and a file separator in the file: the diff keeps "
          "them inside their lines, no marker, and patch accepts it",
          ("\\ No newline" in diffs6[0].text, r.returncode,
           "@@ -2,4 +2,4 @@" in diffs6[0].text),
          (False, 0, True))
    # a long old default is pinned whole in the file and cut in the report
    long_tuple = "(" + ", ".join(f"'k{i}'" for i in range(30)) + ")"
    rows3 = rows + [change(51, "alphalib.core.Client.get",
                           "PARAMETER_CHANGED_DEFAULT", "default",
                           "alphalib.Client.get", long_tuple, "None",
                           score=0.5)]
    proj3 = tmp / "long"
    proj3.mkdir()
    (proj3 / "d.py").write_text('from alphalib import Client\n'
                                'c = Client("h", port=1, retries=3)\n'
                                'v = c.get("k")\n')
    located = loc.locate(proj3, rows3, api)
    plans = rw.decide(located, rows3, api)
    diffs = rw.render(proj3, plans)
    report = rep.report("alphalib", "1.0.0", "2.0.0", [], rows3, located,
                        "d", "/p", [], plans)
    # three shapes of a call: spanning lines without a trailing comma
    # (the pins join the last line and `)` stays), a trailing comma on
    # the same line (one space after each comma), a bare generator
    # argument (it takes parentheses, since it is legal only alone)
    proj4 = tmp / "shapes"
    proj4.mkdir()
    (proj4 / "e.py").write_text('from alphalib import Client\n'
                                'from alphalib.opts import fetch\n'
                                'a = Client(\n    "h",\n    1\n)\n'
                                'b = Client("h",)\n'
                                'c = fetch(x for x in y)\n')
    located4 = loc.locate(proj4, rows, api)
    diffs4 = rw.render(proj4, rw.decide(located4, rows, api))
    check("the pins join a multi-line call's last line, take one space "
          "after a same-line trailing comma, and parenthesise a bare "
          "generator argument",
          diffs4[0].new,
          'from alphalib import Client\n'
          'from alphalib.opts import fetch\n'
          'a = Client(\n    "h",\n    port=1, retries=3\n)\n'
          'b = Client("h", retries=3,)\n'
          'c = fetch((x for x in y), timeout=None, retries=3, verify=True, '
          'pool=None, tags=())\n')
    compile(diffs4[0].new, "e.py", "exec")
    r2line = next(ln for ln in report.splitlines()
                  if "R2, pin the old default: default=(" in ln)
    check("the whole value goes into the diff, and the report's R2 line "
          "is cut with the mark",
          (f'+v = c.get("k", default={long_tuple})' in diffs[0].text,
           long_tuple in r2line, "…" in r2line),
          (True, False, True))


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def case_cli(tmp: pathlib.Path) -> None:
    print("\n5. THE COMMAND LINE: THE REPORT'S DECISIONS, --diff, --apply")
    bundle = tmp / "b.sqlite"
    write_bundle(bundle, all_changes())
    copy = tmp / "cli_app"
    shutil.copytree(PROJECT, copy)
    base = ["alphalib", str(copy), "--from", "1.0.0", "--to", "2.0.0",
            "--bundle", str(bundle), "--env", str(SITE)]
    code, out, err = run(base)
    lines = out.splitlines()
    check("exit 1; the footer counts each rule",
          (code, "R1, pass by name: 16 call(s) rewritten, 9 withheld, "
                 "4 need nothing. R2, pin the old default: 26 call(s) "
                 "pinned, 10 withheld, 2 left to you, 3 need nothing; the "
                 "diff follows." in out), (1, True))
    # the decision prints under the FUNCTION's entry (Client.__init__),
    # not under the class's own row (the base removed), which the same
    # call also meets
    first = next(k for k, ln in enumerate(lines)
                 if "alphalib.core.Client.__init__   used by" in ln)
    i = next(k for k, ln in enumerate(lines)
             if k > first and ln.startswith("       cases.py:9:5 "))
    check("under a rewritten line, the call as it will read, then each "
          "rule's part, under the function's entry",
          (lines[i + 1].strip(), lines[i + 2].strip(), lines[i + 3].strip()),
          ('-> rewritten: Client("localhost", port=6380, db=2, retries=3)',
           "R1, pass by name: port, db",
           "R2, pin the old default: retries=3 (now 5); delete an argument "
           "to accept its new default"))
    j = next(k for k, ln in enumerate(lines)
             if k < first and ln.startswith("       cases.py:9:5 "))
    check("not under the class's entry for the removed base",
          lines[j + 1].startswith("       cases.py:"), True)
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       cases.py:21:5 "))
    check("under a withheld line, the reason and what it would be; no R2 "
          "line where R2 has no row",
          (lines[i + 1].strip(), lines[i + 2].strip(),
           lines[i + 3].strip().startswith("R2")),
          ("-> withheld",
           "R1 withheld: the function also carries a flag: port became "
           "positional-only in 2.0.0; timeout was removed in 2.0.0 | would "
           'be: connect("h", port=2, timeout=9)', False))
    i = next(k for k, ln in enumerate(lines)
             if k > first and ln.startswith("       cases.py:11:5 "))
    check("under a line R1 leaves alone and R2 pins",
          (lines[i + 1].strip(), lines[i + 2].strip(), lines[i + 3].strip()),
          ('-> rewritten: Client("localhost", port=6380, db=2, retries=3)',
           "R1: already passed by name: db, port",
           "R2, pin the old default: retries=3 (now 5); delete an argument "
           "to accept its new default"))
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       cases.py:19:5 "))
    check("under a line that needs nothing, why",
          (lines[i + 1].strip(), lines[i + 2].strip()),
          ("-> needs nothing",
           "R1: the moved parameter(s) are not passed here: timeout"))
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       cases.py:20:5 "))
    check("under helper(1, 2, 3): rewritten by R2, R1 withheld beside it",
          (lines[i + 1].strip(), lines[i + 2].strip(), lines[i + 3].strip()),
          ("-> rewritten: helper(1, 2, 3, d=1)",
           "R1 withheld: b was positional-only in the old version: it has "
           "no name to pass",
           "R2, pin the old default: d=1 (now 2); delete an argument to "
           "accept its new default"))
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       defaults.py:7:5 "))
    check("under fetch(\"u\"): the pins with their new values, what is not "
          "pinned and why, and the cosmetic row",
          (lines[i + 1].strip(), lines[i + 2].strip()),
          ('-> rewritten: fetch("u", timeout=None, retries=3, verify=True, '
           "pool=None, tags=())",
           "R2, pin the old default: timeout=None (now DEFAULT_TIMEOUT), "
           "retries=3 (now 5), verify=True (now False), pool=None (now "
           "Pool()), tags=() (now None); not pinned: limit (old default "
           "DEFAULT_LIMIT is not a plain value; now 20), hooks (old default "
           "[] is a mutable container; now None); changed only in "
           "spelling: mode ('fast' -> \"fast\"); delete an argument to "
           "accept its new default"))
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       defaults.py:9:5 "))
    check("under the all-named fetch: left to you, the rows only the user "
          "can set",
          (lines[i + 1].strip(), lines[i + 2].strip()[:60]),
          ("-> left to you",
           "R2 left to you: limit (old default DEFAULT_LIMIT is not a pl"))
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       defaults.py:33:5 "))
    check("under wrap(1): withheld, y now required and unpinnable, the pin "
          "it would make",
          (lines[i + 1].strip(), lines[i + 2].strip(), lines[i + 3].strip()),
          ("-> withheld",
           "R1: the moved parameter(s) are not passed here: z; flagged: y "
           "is now required in 2.0.0 and cannot be pinned (old default 2 is "
           "positional-only in the old version: no name to pass)",
           "R2 withheld: the function also carries a flag: y is now "
           "required in 2.0.0 and cannot be pinned (old default 2 is "
           "positional-only in the old version: no name to pass) | would "
           "be: wrap(1, z=3)"))
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("       defaults.py:18:5 "))
    check("under ship(\"p\"): withheld by the function's flag, with the pin "
          "it would make",
          (lines[i + 1].strip(), lines[i + 2].strip()),
          ("-> withheld",
           "R2 withheld: the function also carries a flag: dest was added "
           'as required in 2.0.0 | would be: ship("p", compress=False)'))
    check("the diffs are printed after the report, and the files are not "
          "written without --apply",
          ("--- a/cases.py" in out and "+++ b/cases.py" in out,
           "--- a/defaults.py" in out,
           '+a = Client("localhost", port=6380, db=2, retries=3)' in out,
           'a = Client("localhost", 6380, 2)' in (copy / CASES).read_text()),
          (True, True, True, True))
    code, out, err = run(base + ["--diff", str(tmp / "r.diff")])
    check("--diff writes the diff to the path and says so",
          (code, "--- a/cases.py" in out,
           f"diff written to {tmp / 'r.diff'} (2 file(s))" in out,
           "+++ b/defaults.py" in (tmp / "r.diff").read_text()),
          (1, False, True, True))
    r = subprocess.run(["patch", "-p1", "--dry-run", "-i",
                        str(tmp / "r.diff")], cwd=copy, capture_output=True,
                       text=True)
    check("patch -p1 accepts the written diff", r.returncode, 0)
    code, out, err = run(base + ["--apply"])
    check("--apply writes the files, prints the diff and says what it wrote",
          (code, "2 file(s) rewritten: cases.py, defaults.py" in out,
           "--- a/cases.py" in out,
           'a = Client("localhost", port=6380, db=2, retries=3)'
           in (copy / CASES).read_text(),
           'p = fetch(\n    "u",\n    retries=1,\n    timeout=None,\n'
           in (copy / DEFAULTS).read_text()), (1, True, True, True, True))
    code, out, err = run(base)
    check("the run after --apply: nothing to rewrite, no diff, still exit 1 "
          "for the flagged lines",
          (code, "R1, pass by name: 0 call(s) rewritten, 9 withheld, "
                 "20 need nothing. R2, pin the old default: 0 call(s) "
                 "pinned, 10 withheld, 13 left to you, 18 need nothing."
                 in out,
           "--- a/" in out),
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
          (code, "R1, pass by name: 0 call(s) rewritten, 24 withheld, 1 need "
                 "nothing. R2, pin the old default: 0 call(s) pinned, 37 "
                 "withheld, 2 left to you, 1 need nothing." in out,
           "R1 withheld: the old signature could not be read: the old "
           "version was not read (see the environment line above)" in out,
           "R2 withheld: the old signature could not be read: the old "
           "version was not read (see the environment line above)" in out,
           "--- a/" in out), (1, True, True, True, False))


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
        print("\nDo not ship R2 until this passes.")
        sys.exit(1)
    print("All checks passed. R1 names what the old signature names, R2 pins")
    print("the old default where it is a plain value, and both say why")
    print("everywhere else.")


if __name__ == "__main__":
    main()
