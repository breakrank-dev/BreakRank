"""
Does locate find every line of a project that uses a changed name, say how
sure it is of each, and report the rest? (NOTES §38.3, §38.5, §38.2)

    python scripts/test_fixer_locate.py

No database, no network, nothing imported from the fixture: the project
under scripts/fixtures/alpha_app/ names a pretend package alphalib every
way a project can, and scripts/fixtures/site/ holds that package's old
version, 1.0.0, with its dist-info, read with griffe and never imported.
The twelve changes of alphalib 1.0.0 -> 2.0.0 are written here; the
expected hits were written BEFORE the code ran, one per line of the
fixture (file:line:col, the column 1-based as editors show it;
  confidence  use  change ids  path the code names):

  app.py
    9:27   resolved    import     12       alphalib.core._internal
    12:10  resolved    call       1,2,3,10 alphalib.Client   3 positional
    13:1   resolved    method     4        alphalib.Client.send  on Base
    14:9   resolved    call       5        alphalib.connect  2 + timeout=
    15:7   resolved    attribute  7        alphalib.util.LIMIT (as CAP)
    15:13  resolved    attribute  7        alphalib.util.LIMIT (util.LIMIT)
    17:1   resolved    method     9        alphalib.Pool.acquire, the path
                                           resolved through the old version
    18:14  referenced  argument   1,2,3    alphalib.Client in partial()
    18:14  resolved    argument   10       the same name, the base removed
    19:1   resolved    call       8        alphalib.util.ping
    20:7   resolved    argument   12       print(_internal)
    24:9   resolved    call       1,2,3,10 Client(host) in a function
    26:12  resolved    method     4        c.send(payload=) same function
    30:5   unverified  method     4        x.send: x of unknown type
    31:5   unverified  method     9        x.acquire
    32:19  referenced  argument   1,2,3    isinstance(x, Client)
    32:19  resolved    argument   10       the same, the base removed
    33:12  resolved    call       6        alphalib.helper(1, 2, 3)
    36     nothing: `name: Client` is an annotation
    37:12  resolved    call       1,2,3,10 Client("h") made in place
    37:12  resolved    method     4        .send on it, not explicit self
    40:1   referenced  subclass   1,2,3    class MyClient(Client)
    40:1   resolved    subclass   10       the same, the base removed
    42:9   referenced  super      1,2,3    super().__init__(host, 1)
    46:9   resolved    method     9        self.pool.acquire, self.pool set
                                           in __init__ from alphalib.Pool()
    47:16  unverified  method     4        self.send: the project class is
                                           not followed
    50:6   resolved    call       1,2,3,10 with Client("w") as cw
    51:5   resolved    method     4        cw.send(..., timeout=1)
  services/worker.py
    9:9    resolved    call       1,2,3,10 C("h", port=1), C an alias
    10:5   resolved    method     4        Base.send(c, ...) self explicit
    11:5   resolved    method     4        alphalib.core.Client.send(c, ..)
                                           self explicit, found on Base
    13     nothing: s is requests.Session(), another package's object
    18:12  resolved    method     4        worker_client.send, the instance
                                           made at module level below it
    21:17  resolved    call       1,2,3,10 worker_client = C("later")
  services/legacy.py
     2     a note: `from alphalib.util import *`, the tool cannot see
           through it; ping("h") on line 6 is therefore not a hit

  32 hits on 27 lines: 25 resolved, 4 referenced, 3 unverified; 4 files
  read; change
  11 (RATE) untouched, the other eleven touched; change 4 has 9 hits,
  changes 1, 2, 3 have 10 each, change 10 has 9, change 9 has 3.

  Without the old version (its bases unread, --from given but not
  installed): the same 32 lines, 17 resolved, 4 referenced, 11
  unverified: the eight method calls on Client and Pool instances whose
  path the bundle does not name (13, 17, 26, 37's .send, 46, 51, worker
  11 and 18) become unverified, and Base.send(c, ...) stays resolved
  since the bundle names that path.

The report: PART 1 lists the nine touched symbols in the site's order
(the six at score 1.0 by name, Client, connect, helper, LIMIT, ping; then
Base.send, Client.__init__ with its three rows, _internal marked private,
Pool.acquire), each with "used by" and its lines; PART 2 lists RATE alone.
The command line: --from read from the environment's dist-info, exit 1
with hits, 0 with none, 2 when it cannot run.
"""

import contextlib
import io
import pathlib
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fixer.bundle as fb  # noqa: E402
import fixer.cli as cli  # noqa: E402
import fixer.env as env  # noqa: E402
import fixer.locate as loc  # noqa: E402
import fixer.oldapi as oldapi  # noqa: E402
import fixer.report as rep  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
PROJECT = FIXTURES / "alpha_app"
SITE = FIXTURES / "site"

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


# ------------------------------------------------------------------ fixture

def change(cid, symbol, kind, sub="", exports="", old=None, new=None,
           private=False, score=1.0, used=0, depth=None):
    """One bundle row for alphalib 1.0.0 -> 2.0.0."""
    what = symbol.rsplit(".", 1)[-1] + (f"({sub})" if sub else "")
    message = f"{what}: {kind.replace('_', ' ').lower()}"
    if old is not None:
        message += f" was changed: {old} -> {new}"
    return fb.Change(cid, "alphalib", "1.0.0", "2.0.0", symbol,
                     tuple(p for p in exports.split(";") if p), kind, sub,
                     message, old, new, private, symbol.count("."),
                     depth if depth is not None else symbol.count("."),
                     0, used, used, score)


def changes() -> list:
    init = "alphalib.core.Client.__init__"
    return [
        change(1, init, "PARAMETER_MOVED", "port",
               "alphalib.Client.__init__", score=0.5, used=7),
        change(2, init, "PARAMETER_MOVED", "db",
               "alphalib.Client.__init__", score=0.5, used=7),
        change(3, init, "PARAMETER_CHANGED_DEFAULT", "retries",
               "alphalib.Client.__init__", "3", "5", score=0.5, used=7),
        change(4, "alphalib.core.Base.send", "PARAMETER_CHANGED_KIND",
               "timeout", "alphalib.Base.send", score=0.5, used=5),
        change(5, "alphalib.core.connect", "PARAMETER_REMOVED", "timeout",
               "alphalib.connect", used=2),
        change(6, "alphalib.core.helper", "PARAMETER_CHANGED_DEFAULT", "d",
               "alphalib.helper", "1", "2", used=1),
        change(7, "alphalib.util.LIMIT", "ATTRIBUTE_CHANGED_VALUE", "",
               "alphalib.LIMIT", "10", "20", used=3),
        change(8, "alphalib.util.ping", "OBJECT_REMOVED", "",
               "alphalib.ping", used=4),
        change(9, "alphalib.util.Pool.acquire", "PARAMETER_ADDED_REQUIRED",
               "timeout", "", score=0.5),
        change(10, "alphalib.core.Client", "CLASS_REMOVED_BASE", "Base",
               "alphalib.Client", used=9),
        change(11, "alphalib.util.RATE", "ATTRIBUTE_CHANGED_VALUE", "",
               "alphalib.RATE", "0.5", "0.25"),
        change(12, "alphalib.core._internal", "OBJECT_REMOVED", "", "",
               private=True, score=0.5),
    ]


def extra_changes() -> list:
    """Three more, for the corners the review of 20 Oct named: a removed
    MODULE, a symbol whose export path is under another import root, and
    a classmethod."""
    return [
        change(13, "alphalib.compat", "OBJECT_REMOVED"),
        change(14, "_alphalib.impl.compute", "PARAMETER_REMOVED", "x",
               "alphalib.compute"),
        change(15, "alphalib.core.Client.from_url", "PARAMETER_MOVED",
               "retries", "alphalib.Client.from_url", score=0.5),
    ]


def row(file, line, col, confidence, use, ids, path, receiver=None,
        positional=0, keywords=(), star=False, explicit=False):
    return (file, line, col, confidence, use, tuple(ids), path, receiver,
            positional, tuple(keywords), star, explicit)


def as_row(h: loc.Hit):
    return (h.file, h.line, h.column, h.confidence, h.use, h.change_ids,
            h.path, h.receiver, h.positional, h.keywords, h.star_args,
            h.explicit_self)


APP, WORKER = "app.py", "services/worker.py"
CTOR = (1, 2, 3, 10)
BASE, CLIENT, POOL = ("alphalib.core.Base", "alphalib.core.Client",
                      "alphalib.util.Pool")

EXPECTED_WITH_OLD = [
    row(APP, 9, 27, "resolved", "import", (12,), "alphalib.core._internal"),
    row(APP, 12, 10, "resolved", "call", CTOR, "alphalib.Client",
        CLIENT, 3),
    row(APP, 13, 1, "resolved", "method", (4,), "alphalib.Client.send", BASE,
        2),
    row(APP, 14, 9, "resolved", "call", (5,), "alphalib.connect", None, 2,
        ("timeout",)),
    row(APP, 15, 7, "resolved", "attribute", (7,), "alphalib.util.LIMIT"),
    row(APP, 15, 13, "resolved", "attribute", (7,), "alphalib.util.LIMIT"),
    row(APP, 17, 1, "resolved", "method", (9,), "alphalib.Pool.acquire",
        POOL),
    row(APP, 18, 14, "referenced", "argument", (1, 2, 3), "alphalib.Client"),
    row(APP, 18, 14, "resolved", "argument", (10,), "alphalib.Client"),
    row(APP, 19, 1, "resolved", "call", (8,), "alphalib.util.ping", None, 2),
    row(APP, 20, 7, "resolved", "argument", (12,), "alphalib.core._internal"),
    row(APP, 24, 9, "resolved", "call", CTOR, "alphalib.Client",
        CLIENT, 1),
    row(APP, 26, 12, "resolved", "method", (4,), "alphalib.Client.send", BASE,
        0, ("payload",)),
    row(APP, 30, 5, "unverified", "method", (4,), "?.send", None, 1),
    row(APP, 31, 5, "unverified", "method", (9,), "?.acquire"),
    row(APP, 32, 19, "referenced", "argument", (1, 2, 3), "alphalib.Client"),
    row(APP, 32, 19, "resolved", "argument", (10,), "alphalib.Client"),
    row(APP, 33, 12, "resolved", "call", (6,), "alphalib.helper", None, 3),
    row(APP, 37, 12, "resolved", "call", CTOR, "alphalib.Client",
        CLIENT, 1),
    row(APP, 37, 12, "resolved", "method", (4,), "alphalib.Client.send", BASE,
        1),
    row(APP, 40, 1, "referenced", "subclass", (1, 2, 3), "alphalib.Client"),
    row(APP, 40, 1, "resolved", "subclass", (10,), "alphalib.Client"),
    row(APP, 42, 9, "referenced", "super", (1, 2, 3),
        "alphalib.Client.__init__", CLIENT, 2),
    row(APP, 46, 9, "resolved", "method", (9,), "alphalib.Pool.acquire",
        POOL, 1),
    row(APP, 47, 16, "unverified", "method", (4,), "?.send", None, 2),
    row(APP, 50, 6, "resolved", "call", CTOR, "alphalib.Client",
        CLIENT, 1),
    row(APP, 51, 5, "resolved", "method", (4,), "alphalib.Client.send", BASE,
        1, ("timeout",)),
    row(WORKER, 9, 9, "resolved", "call", CTOR, CLIENT, CLIENT, 1,
        ("port",)),
    row(WORKER, 10, 5, "resolved", "method", (4,), "alphalib.core.Base.send",
        BASE, 3, explicit=True),
    row(WORKER, 11, 5, "resolved", "method", (4,),
        "alphalib.core.Client.send", BASE, 2, explicit=True),
    row(WORKER, 18, 12, "resolved", "method", (4,),
        "alphalib.core.Client.send", BASE, 1),
    row(WORKER, 21, 17, "resolved", "call", CTOR, CLIENT, CLIENT, 1),
]

# Without the old version: the receivers are known by name only, so the
# method calls whose path the bundle does not name are unverified, with
# the path as the code names it; the rest is unchanged.
UNVERIFIED_WITHOUT_OLD = {
    (APP, 13): row(APP, 13, 1, "unverified", "method", (4,),
                   "alphalib.Client.send", None, 2),
    (APP, 17): row(APP, 17, 1, "unverified", "method", (9,), "?.acquire"),
    (APP, 26): row(APP, 26, 12, "unverified", "method", (4,),
                   "alphalib.Client.send", None, 0, ("payload",)),
    (APP, 37): row(APP, 37, 12, "unverified", "method", (4,),
                   "alphalib.Client.send", None, 1),
    (APP, 46): row(APP, 46, 9, "unverified", "method", (9,), "?.acquire",
                   None, 1),
    (APP, 51): row(APP, 51, 5, "unverified", "method", (4,),
                   "alphalib.Client.send", None, 1, ("timeout",)),
    (WORKER, 11): row(WORKER, 11, 5, "unverified", "method", (4,),
                      "alphalib.core.Client.send", None, 2),
    (WORKER, 18): row(WORKER, 18, 12, "unverified", "method", (4,),
                      "alphalib.core.Client.send", None, 1),
}


def expected_without_old():
    out = []
    for r in EXPECTED_WITH_OLD:
        key = (r[0], r[1])
        if key in UNVERIFIED_WITHOUT_OLD and r[4] == "method":
            out.append(UNVERIFIED_WITHOUT_OLD[key])
        elif key == (WORKER, 10):
            # Base is not known to be a class without the old version, so
            # the call is recorded as a call, with no self to offset
            out.append(row(WORKER, 10, 5, "resolved", "call", (4,),
                           "alphalib.core.Base.send", None, 3))
        elif r[7] == CLIENT and r[6].startswith("alphalib.Client"):
            # the receiver is the class as the code names it
            out.append(r[:7] + ("alphalib.Client",) + r[8:])
        else:
            out.append(r)
    return sorted(out, key=lambda r: (r[0], r[1], r[2], r[4], r[3]))


def compare_rows(name, got, want) -> None:
    check(name, got, want)
    if got != want:
        for g in got:
            if g not in want:
                print(f"          extra    {g!r}")
        for w in want:
            if w not in got:
                print(f"          missing  {w!r}")


# -------------------------------------------------------------- 1. env

def case_env(tmp: pathlib.Path) -> None:
    print("\n1. THE ENVIRONMENT: DIST-INFO READ FROM DISK, NOTHING IMPORTED")
    check("names normalise as PEP 503 with _ as the dist-info spelling",
          [env.normalise(n) for n in ("Databricks-SDK", "a.b_c-d",
                                      "zope.interface", "ALPHALIB")],
          ["databricks_sdk", "a_b_c_d", "zope_interface", "alphalib"])
    got = env.find_installed("alphalib", SITE)
    check("the fixture's alphalib: name, version and import root from "
          "METADATA and top_level.txt",
          (got.distribution, got.version, got.import_roots,
           got.dist_info.name),
          ("alphalib", "1.0.0", ("alphalib",), "alphalib-1.0.0.dist-info"))
    check("the environment given as the directory holding the dist-info",
          env.site_packages(SITE), SITE)
    try:
        env.find_installed("alpha-lib", SITE)
        got = "no error"
    except env.EnvError as e:
        got = ("not installed in" in str(e)
               and "alpha_lib-*.dist-info" in str(e))
    check("a distribution not there: EnvError naming the directory and "
          "the dist-info looked for", got, True)

    venv = tmp / "venv"
    sp = venv / "lib" / "python3.12" / "site-packages"
    di = sp / "beta_lib-2.0.dist-info"
    di.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: Beta-Lib\n"
                                 "Version: 2.0\n\nBody: not read\n")
    (di / "RECORD").write_text(
        "betalib/__init__.py,sha256=abc,12\n"
        "betalib/core.py,sha256=abc,40\n"
        "betalib/__pycache__/core.cpython-312.pyc,,\n"
        "beta_lib-2.0.dist-info/METADATA,sha256=abc,99\n"
        "../../../bin/beta,sha256=abc,10\n"
        "_betalib_speedups.cpython-312-x86_64-linux-gnu.so,,\n"
        "betalib_single.py,sha256=abc,5\n"
        "__pycache__/betalib_single.cpython-312.pyc,,\n")
    got = env.find_installed("beta.lib", venv)
    check("a venv root: the dist-info found under lib/python3.*/site-"
          "packages, the name normalised both ways, the roots from RECORD "
          "without the dist-info, caches, scripts or compiled modules",
          (got.distribution, got.version, got.import_roots),
          ("Beta-Lib", "2.0", ("betalib", "betalib_single")))
    check("the same through the site-packages directory itself",
          env.find_installed("BETA_LIB", sp).version, "2.0")
    (sp / "alphalib-9.9.9.dist-info").mkdir()
    (sp / "alphalib-9.9.9.dist-info" / "METADATA").write_text(
        "Name: otherlib\nVersion: 9.9.9\n")
    try:
        env.find_installed("alphalib", sp)
        got = "no error"
    except env.EnvError as e:
        got = "not installed in" in str(e)
    check("a dist-info whose directory name says alphalib but whose METADATA "
          "names another distribution is not alphalib", got, True)
    (sp / "beta_lib-2.1.dist-info").mkdir()
    (sp / "beta_lib-2.1.dist-info" / "METADATA").write_text(
        "Name: beta-lib\nVersion: 2.1\n")
    try:
        env.find_installed("beta-lib", sp)
        got = "no error"
    except env.EnvError as e:
        got = "more than once" in str(e) and "2.0, 2.1" in str(e)
    check("two dist-infos of one distribution: EnvError with both versions",
          got, True)
    (tmp / "empty").mkdir()
    for path, want in ((tmp / "empty", "neither a directory of installed"),
                       (tmp / "missing", "is not a directory")):
        try:
            env.site_packages(path)
            got = "no error"
        except env.EnvError as e:
            got = want in str(e)
        check(f"--env {path.name}: EnvError, {want!r}", got, True)
    check("no --env: the running interpreter's site-packages, which exists",
          env.site_packages(None).is_dir(), True)


# ----------------------------------------------------------- 2. old api

def case_oldapi() -> None:
    print("\n2. THE OLD VERSION, READ WITH GRIFFE: ALIASES, BASES, PARAMETERS")
    api = oldapi.OldApi(("alphalib",), SITE)
    check("alphalib loads statically, no errors", (api.loaded, api.errors),
          (True, {}))
    check("an export path resolves to the defining path",
          api.canonical("alphalib.Client"), "alphalib.core.Client")
    check("a method reached through the subclass resolves to the base that "
          "defines it", api.canonical("alphalib.Client.send"),
          "alphalib.core.Base.send")
    check("the MRO, nearest first, by defining paths",
          api.mro("alphalib.Client"),
          ["alphalib.core.Client", "alphalib.core.Base"])
    check("which class defines a member: send on Base, get on Client, an "
          "unknown member on none",
          (api.defining_class("alphalib.Client", "send"),
           api.defining_class("alphalib.Client", "get"),
           api.defining_class("alphalib.Client", "nothing")),
          ("alphalib.core.Base", "alphalib.core.Client", None))
    P = oldapi.Param
    check("parameters with their kinds and defaults, in order",
          api.parameters("alphalib.core.helper"),
          [P("a", "positional_only", None), P("b", "positional_only", None),
           P("c", "positional_or_keyword", None),
           P("d", "keyword_only", "1")])
    check("a class: its __init__'s parameters, self included",
          api.parameters("alphalib.Client"),
          [P("self", "positional_or_keyword", None),
           P("host", "positional_or_keyword", None),
           P("port", "positional_or_keyword", "6379"),
           P("db", "positional_or_keyword", "0"),
           P("retries", "positional_or_keyword", "3")])
    check("not a function, a class without __init__, an unknown name, "
          "another package: None each",
          (api.parameters("alphalib.LIMIT"), api.parameters("alphalib.Pool"),
           api.parameters("alphalib.nothing"), api.lookup("other.x")),
          (None, None, None, None))
    check("a class that inherits __init__ answers with the base's",
          api.parameters("alphalib.core.Child"),
          api.parameters("alphalib.Client"))
    check("labels: a classmethod is one, a method is not, an unknown name "
          "has none",
          (api.labels("alphalib.Client.from_url"),
           api.labels("alphalib.Client.send"), api.labels("alphalib.zzz")),
          ({"classmethod"}, set(), set()))
    check("a path through another package's import (compat does `import "
          "json`) is None, not an exception",
          (api.lookup("alphalib.compat.json.dumps"),
           api.canonical("alphalib.compat.json"),
           api.lookup("alphalib.compat.old_ping").path),
          (None, None, "alphalib.compat.old_ping"))
    missing = oldapi.OldApi(("nosuchpkg",), SITE)
    check("a root griffe cannot find: not loaded, the error kept, every "
          "question None",
          (missing.loaded, list(missing.errors), missing.canonical("x.y"),
           missing.mro("x.Y"), missing.parameters("x.f")),
          (False, ["nosuchpkg"], None, [], None))
    check("no site: nothing loaded, no error",
          (oldapi.OldApi(("alphalib",), None).loaded,
           oldapi.OldApi(("alphalib",), None).errors), (False, {}))


# ------------------------------------------------------------ 3. locate

def case_locate() -> loc.Located:
    print("\n3. LOCATE ON THE FIXTURE PROJECT, THE OLD VERSION READ")
    api = oldapi.OldApi(("alphalib",), SITE)
    got = loc.locate(PROJECT, changes(), api)
    rows = [as_row(h) for h in got.hits]
    compare_rows("every expected hit and no other, with its confidence, use, "
                 "change ids, path, receiver and call shape",
                 rows, EXPECTED_WITH_OLD)
    check("32 hits: 25 resolved, 4 referenced, 3 unverified; 4 files read, "
          "3 mention alphalib (services/__init__.py does not)",
          (len(got.hits), got.counts(), got.files_read, got.files_parsed),
          (32, {"resolved": 25, "referenced": 4, "unverified": 3}, 4, 3))
    by = got.by_change()
    check("hits per change: 9 on Base.send, 10 on each __init__ row, 9 on "
          "the class, 3 on Pool.acquire, 2 on LIMIT and _internal, none "
          "on RATE",
          {k: len(v) for k, v in sorted(by.items())},
          {1: 10, 2: 10, 3: 10, 4: 9, 5: 1, 6: 1, 7: 2, 8: 1, 9: 3, 10: 9,
           12: 2})
    check("the star import is a note on its file, and its ping() no hit",
          [(n.file, n.text) for n in got.notes],
          [("services/legacy.py", "line 2: `from alphalib.util import *`: "
            "the tool cannot see which names this file uses through it")])
    check("the source line travels with the hit",
          [h.source for h in got.hits if h.line == 42 and h.file == APP],
          ["super().__init__(host, 1)"])
    return got


def case_locate_without_old() -> None:
    print("\n4. THE SAME PROJECT, THE OLD VERSION NOT READ")
    got = loc.locate(PROJECT, changes(), None)
    rows = [as_row(h) for h in got.hits]
    compare_rows("the eight instance-method calls the bundle cannot name "
                 "become unverified with the path as written, Base.send(c, "
                 "...) is a plain call since Base is not known to be a "
                 "class, receivers are the class as written; the rest "
                 "holds", rows, expected_without_old())
    check("32 hits: 17 resolved, 4 referenced, 11 unverified",
          (len(got.hits), got.counts()),
          (32, {"resolved": 17, "referenced": 4, "unverified": 11}))
    unloaded = oldapi.OldApi(("nosuchpkg",), SITE)
    check("an OldApi that loaded nothing counts as not read",
          loc.locate(PROJECT, changes(), unloaded).counts(),
          {"resolved": 17, "referenced": 4, "unverified": 11})


def case_corners(tmp: pathlib.Path) -> None:
    print("\n5. CORNERS: SKIPPED DIRECTORIES, UNREADABLE FILES, SCOPES, "
          "RELATIVE IMPORTS")
    proj = tmp / "corners"
    for d in (".venv/lib/python3.12/site-packages", "node_modules/x",
              "build/lib", "pkg"):
        (proj / d).mkdir(parents=True)
    hidden = "import alphalib\nalphalib.util.ping()\n"
    (proj / ".venv/lib/python3.12/site-packages/x.py").write_text(hidden)
    (proj / "node_modules/x/y.py").write_text(hidden)
    (proj / "build/lib/z.py").write_text(hidden)
    (proj / "bad_bytes.py").write_bytes(b"x = '\xff\xfe'\n")
    (proj / "broken.py").write_text("import alphalib\ndef (:\n    pass\n")
    (proj / "plain.py").write_text(
        "# mentions alphalib in a comment, imports nothing of it\n"
        "def f(x):\n"
        "    return x.send(1)\n")
    (proj / "pkg" / "__init__.py").write_text("")
    (proj / "pkg" / "other.py").write_text(
        "from alphalib import Client\nthing = Client('h')\n")
    (proj / "pkg" / "uses.py").write_text(
        "from .other import thing\n"
        "import alphalib as al\n"
        "from alphalib import Client as K, helper, LIMIT, Pool\n"
        "import requests\n"
        "\n"
        "thing.send(1)\n"
        "al.Client('h')\n"
        "K('x', *rest, **kw)\n"
        "helper(1, 2, c=3, d=4)\n"
        "print(LIMIT)\n"
        "r = requests.get('u')\n"
        "r.send(1)\n"
        "requests.Session().send(1)\n"
        "c = K('h')\n"
        "c = make()\n"
        "c.send(1)\n"
        "getattr(al, 'ping')()\n"
        "\n"
        "\n"
        "def f(a: K) -> K:\n"
        "    return a\n"
        "\n"
        "\n"
        "def outer():\n"
        "    inner_client = K('h')\n"
        "\n"
        "    def inner():\n"
        "        return inner_client.send(2)\n"
        "    return inner\n"
        "\n"
        "\n"
        "class A:\n"
        "    def run(self):\n"
        "        return self.c.send(3)\n"
        "\n"
        "    def __init__(self):\n"
        "        self.c = K('h')\n"
        "        K.__init__(self, 'h')\n"
        "\n"
        "\n"
        "class B(al.Client):\n"
        "    def send(self, payload, timeout=10):\n"
        "        return super().send(payload)\n"
        "\n"
        "\n"
        "pool: Pool = Pool()\n"
        "pool.acquire()\n"
        "s = 'x'\n"
        "s.send(1)\n"
        "c2 = K('h')\n"
        "c3 = c2\n"
        "c3.send(2)\n"
        "k = al.connect('h')\n"
        "k.send(3)\n"
        "def ctx():\n"
        "    with K('h') as cw2:\n"
        "        return cw2.send(4)\n"
        "al.connect('h').send(5)\n"
        "class D:\n"
        "    shared = K('h')\n"
        "\n"
        "    def go(self):\n"
        "        return self.shared.send(6)\n")
    api = oldapi.OldApi(("alphalib",), SITE)
    got = loc.locate(proj, changes(), api)
    check("files under .venv, node_modules and build are not read; the six "
          "others are, and the four that mention alphalib are parsed; "
          "plain.py's x.send(1) is not listed, since the file imports "
          "nothing of alphalib",
          (got.files_read, got.files_parsed,
           [h for h in got.hits if h.file == "plain.py"]), (6, 4, []))
    check("a file not in UTF-8 and a file that does not parse are notes, "
          "not failures",
          sorted((n.file, n.text.split(":")[0]) for n in got.notes),
          [("bad_bytes.py", "not UTF-8; not read"),
           ("broken.py", "not parsed")])
    rows = [as_row(h) for h in got.hits if h.file == "pkg/uses.py"]
    U = "pkg/uses.py"
    want = [
        row(U, 6, 1, "unverified", "method", (4,), ".other.thing.send", None,
            1),
        row(U, 7, 1, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 8, 1, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1, (), True),
        row(U, 9, 1, "resolved", "call", (6,), "alphalib.helper", None, 2,
            ("c", "d")),
        row(U, 10, 7, "resolved", "argument", (7,), "alphalib.LIMIT"),
        row(U, 14, 5, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 16, 1, "unverified", "method", (4,), "?.send", None, 1),
        row(U, 25, 20, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 28, 16, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 1),
        row(U, 34, 16, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 1),
        row(U, 37, 18, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 38, 9, "resolved", "method", (1, 2, 3),
            "alphalib.Client.__init__", CLIENT, 2, explicit=True),
        row(U, 41, 1, "referenced", "subclass", (1, 2, 3), "alphalib.Client"),
        row(U, 41, 1, "resolved", "subclass", (10,), "alphalib.Client"),
        row(U, 43, 16, "referenced", "super", (4,), "alphalib.Client.send",
            BASE, 1),
        row(U, 47, 1, "resolved", "method", (9,), "alphalib.Pool.acquire",
            POOL),
        row(U, 50, 6, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 52, 1, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 1),
        row(U, 53, 5, "resolved", "call", (5,), "alphalib.connect", None, 1),
        row(U, 54, 1, "unverified", "method", (4,), "?.send", None, 1),
        row(U, 56, 10, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 57, 16, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 1),
        row(U, 58, 1, "resolved", "call", (5,), "alphalib.connect", None, 1),
        row(U, 58, 1, "unverified", "method", (4,), "alphalib.connect.send",
            None, 1),
        row(U, 60, 14, "resolved", "call", CTOR, "alphalib.Client",
            CLIENT, 1),
        row(U, 63, 16, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 1),
    ]
    compare_rows("a relative import is unverified, not another package's; "
                 "an alias of the package, a star-argument call, keywords, "
                 "a constant as argument, a variable rebound by an unknown "
                 "call then used, "
                 "another package's objects dropped, getattr invisible, an "
                 "annotation ignored, a nested function's closure, self.c "
                 "set after the method that uses it, Class.__init__(self), "
                 "an attribute base class, super().send, an annotated "
                 "assignment, a string's .send not a hit, a name bound to "
                 "another name's instance, the result of the package's own "
                 "function unverified, `with` inside a function, a method "
                 "on the result of the package's function, a class "
                 "attribute used as self.shared", rows, want)
    check("pkg/other.py: the one call", [as_row(h) for h in got.hits
                                         if h.file == "pkg/other.py"],
          [row("pkg/other.py", 2, 9, "resolved", "call", CTOR,
               "alphalib.Client", CLIENT, 1)])
    try:
        loc.locate(tmp / "nowhere", changes(), None)
        got2 = "no error"
    except FileNotFoundError as e:
        got2 = "not a directory" in str(e)
    check("a project path that is not a directory raises", got2, True)
    empty = tmp / "nothing"
    empty.mkdir()
    (empty / "x.py").write_text("x = 1\n")
    got3 = loc.locate(empty, changes(), None)
    check("a project that names nothing: no hits, no notes, one file read",
          (got3.hits, got3.notes, got3.files_read), ([], [], 1))


def case_review(tmp: pathlib.Path) -> None:
    print("\n5b. THE REVIEW'S CORNERS: SHADOWING, THE PROJECT'S OWN MODULES, "
          "ANOTHER ROOT, A REMOVED MODULE, A CLASSMETHOD, A FILE LIBCST "
          "CANNOT ANALYSE, SKIPPED DIRECTORIES, A WALRUS, A CONSTANT AS "
          "RECEIVER, A LOCAL DEF HIDING AN IMPORT")
    proj = tmp / "review"
    app = proj / "myapp"
    for d in ("myapp/build", "build", "myenv", "lib/myapp2"):
        (proj / d).mkdir(parents=True)
    (proj / "lib/myapp2/factory.py").write_text(
        "from alphalib import Client\n\n\ndef make(h):\n"
        "    return Client(h)\n")
    (proj / "lib/myapp2/handlers.py").write_text(
        "import alphalib\nfrom myapp2.factory import make\nc = make('h')\n"
        "c.send(b'a', 1)\n")
    (proj / "lib/myapp2/only.py").write_text(
        "from myapp2.factory import make\nmake('x')\n")
    (app / "comment.py").write_text(
        "# uses alphalib through the project's wrap module only\n"
        "from myapp.wrap import Client\nc = Client('h')\nc.send(b'w', 1)\n")
    (app / "more.py").write_text(
        "import alphalib\n"
        "from alphalib import Base, Client, LIMIT\n"
        "if (w := Client('w')):\n"
        "    w.send(b'walrus', 1)\n"
        "b = Base()\n"
        "b.send(b'base', 1)\n"
        "n = alphalib.LIMIT.bit_length()\n"
        "m = LIMIT.real\n"
        "\n"
        "\n"
        "def Client(h):\n"
        "    return h\n"
        "\n"
        "\n"
        "Client('shadowed')\n")
    (app / "__init__.py").write_text("")
    (app / "factory.py").write_text(
        "from alphalib import Client\n\n\ndef make_client(h):\n"
        "    return Client(h)\n")
    (app / "wrap.py").write_text("from alphalib import Client, connect\n")
    (app / "consumer.py").write_text(
        "from myapp.wrap import Client, connect\n\nc = Client('h', 1, 2)\n"
        "c.send(b'x', 1)\nconnect('h', 1, timeout=3)\n")
    (app / "shadow.py").write_text(
        "import alphalib\n"
        "from alphalib import Client\n"
        "from myapp.factory import make_client\n"
        "from .factory import make_client as rel_make\n"
        "\n"
        "client = Client('m')\n"
        "pool = alphalib.Pool()\n"
        "\n"
        "\n"
        "def f(client):\n"
        "    return client.send(b'p', 1)\n"
        "\n"
        "\n"
        "def h(items):\n"
        "    for client in items:\n"
        "        client.send(b'l', 1)\n"
        "    return [client.send(b'c', 1) for client in items]\n"
        "\n"
        "\n"
        "def g(path):\n"
        "    with open(path) as client:\n"
        "        client.send(b'w', 1)\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as pool:\n"
        "        pool.acquire()\n"
        "    fn = lambda client: client.send(b'z', 1)\n"
        "    return fn\n"
        "\n"
        "\n"
        "def k(self_, pool):\n"
        "    return pool.acquire(True)\n"
        "\n"
        "\n"
        "def n():\n"
        "    def client():\n"
        "        return 1\n"
        "    return client.send(b'd', 1)\n"
        "\n"
        "\n"
        "def uses_module_level():\n"
        "    client.send(b'ok', 1)\n"
        "    return pool.acquire()\n"
        "\n"
        "\n"
        "c = make_client('h')\n"
        "c.send(b'abs', 1)\n"
        "r = rel_make('h')\n"
        "r.send(b'rel', 1)\n"
        "\n"
        "\n"
        "class Outer:\n"
        "    def __init__(self):\n"
        "        self.c = Client('o')\n"
        "\n"
        "    class Inner:\n"
        "        def go(self):\n"
        "            return self.c.send(b'i', 1)\n"
        "\n"
        "    def run(self):\n"
        "        return self.c.send(b'r', 1)\n"
        "\n"
        "\n"
        "def m(pools):\n"
        "    client = pools[0]\n"
        "    return client.send(b's', 1)\n"
        "\n"
        "\n"
        "class Shadowing:\n"
        "    pool = 'not the module pool'\n"
        "\n"
        "    def run(self):\n"
        "        return pool.acquire(True)\n")
    (app / "cross.py").write_text(
        "from alphalib import compute\nimport _alphalib.impl\n"
        "compute(1)\n_alphalib.impl.compute(2)\n")
    (app / "gone.py").write_text(
        "import alphalib.compat\n"
        "import alphalib.compat as cp\n"
        "from alphalib.compat import old_ping\n"
        "alphalib.compat.old_ping('x')\n"
        "cp.old_ping('y')\n"
        "old_ping('z')\n"
        "alphalib.compat.json.dumps({})\n")
    (app / "cm.py").write_text(
        "from alphalib import Client\nClient.from_url('u', 5)\n"
        "c = Client.from_url('v')\nc.send(b'q', 1)\n")
    (app / "gen.py").write_text(
        "import alphalib\nTABLE = " + " + ".join(["1"] * 800)
        + "\nalphalib.util.ping('h')\n")
    (app / "build" / "inner.py").write_text(
        "import alphalib\nalphalib.util.ping('h')\n")
    (proj / "build" / "x.py").write_text(
        "import alphalib\nalphalib.util.ping('top')\n")
    (proj / "myenv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (proj / "myenv" / "x.py").write_text(
        "import alphalib\nalphalib.util.ping('venv')\n")
    api = oldapi.OldApi(("alphalib",), SITE)
    got = loc.locate(proj, changes() + extra_changes(), api)
    check("15 files read, 12 parsed (consumer.py, only.py and __init__.py "
          "never mention alphalib), 3 reach the package only through the "
          "project's own modules (two unparsed, comment.py parsed but "
          "importing none of alphalib), build/ at the top and the venv "
          "skipped and named, myapp/build/ read, both import roots known",
          (got.files_read, got.files_parsed, got.files_via_project,
           got.skipped_dirs, got.roots),
          (15, 12, 3, ("build", "myenv"), ("_alphalib", "alphalib")))
    check("a file libcst cannot analyse is a note, not a crash",
          [(n.file, n.text.split(":")[0], "RecursionError" in n.text)
           for n in got.notes],
          [("myapp/gen.py", "not analysed", True)])
    SH = "myapp/shadow.py"
    want = [
        row(SH, 6, 10, "resolved", "call", CTOR, "alphalib.Client", CLIENT,
            1),
        row(SH, 11, 12, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 16, 9, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 17, 13, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 22, 9, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 26, 9, "unverified", "method", (9,), "?.acquire"),
        row(SH, 27, 25, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 32, 12, "unverified", "method", (9,), "?.acquire", None, 1),
        row(SH, 38, 12, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 42, 5, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 2),
        row(SH, 43, 12, "resolved", "method", (9,), "alphalib.Pool.acquire",
            POOL),
        row(SH, 47, 1, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 49, 1, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 54, 18, "resolved", "call", CTOR, "alphalib.Client", CLIENT,
            1),
        row(SH, 58, 20, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 61, 16, "resolved", "method", (4,), "alphalib.Client.send",
            BASE, 2),
        row(SH, 66, 12, "unverified", "method", (4,), "?.send", None, 2),
        row(SH, 73, 16, "resolved", "method", (9,), "alphalib.Pool.acquire",
            POOL, 1),
    ]
    compare_rows("a parameter, a loop variable, a comprehension variable, "
                 "`with ... as`, `except ... as`, a lambda parameter and a "
                 "local def named like a module-level instance are unknown "
                 "in their scope (unverified), the module-level instance "
                 "itself resolved where unshadowed; the result of a project "
                 "function, imported absolutely or relatively, is "
                 "unverified, not dropped; a nested class's self.c is not "
                 "the outer class's; a name rebound to an index is unknown; "
                 "a class body's name is not visible in its methods, the "
                 "module's is",
                 [as_row(h) for h in got.hits if h.file == SH], want)
    only = loc.locate(proj, [extra_changes()[1]], None)
    check("with only the cross-root change, the export path's root counts "
          "as the package's: cross.py is parsed and both spellings found",
          (only.roots, only.files_parsed, [h.line for h in only.hits]),
          (("_alphalib", "alphalib"), 12, [3, 4]))
    CR, GO, CM = "myapp/cross.py", "myapp/gone.py", "myapp/cm.py"
    compare_rows("an export path under another root (_alphalib -> "
                 "alphalib.compute) is found by both spellings",
                 [as_row(h) for h in got.hits if h.file == CR],
                 [row(CR, 3, 1, "resolved", "call", (14,), "alphalib.compute",
                      None, 1),
                  row(CR, 4, 1, "resolved", "call", (14,),
                      "_alphalib.impl.compute", None, 1)])
    compare_rows("a removed module is a hit in every path through it: "
                 "import, import as, from-import, a call through it under "
                 "three spellings, and a path griffe cannot resolve",
                 [as_row(h) for h in got.hits if h.file == GO],
                 [row(GO, 1, 8, "resolved", "import", (13,),
                      "alphalib.compat"),
                  row(GO, 2, 8, "resolved", "import", (13,),
                      "alphalib.compat"),
                  row(GO, 3, 1, "resolved", "import", (13,),
                      "alphalib.compat"),
                  row(GO, 4, 1, "resolved", "attribute", (13,),
                      "alphalib.compat"),
                  row(GO, 5, 1, "resolved", "attribute", (13,),
                      "alphalib.compat"),
                  row(GO, 6, 1, "resolved", "attribute", (13,),
                      "alphalib.compat"),
                  row(GO, 7, 1, "resolved", "attribute", (13,),
                      "alphalib.compat")])
    compare_rows("a classmethod called on the class is not an explicit-self "
                 "call, and its result is an unknown instance",
                 [as_row(h) for h in got.hits if h.file == CM],
                 [row(CM, 2, 1, "resolved", "method", (15,),
                      "alphalib.Client.from_url", CLIENT, 2),
                  row(CM, 3, 5, "resolved", "method", (15,),
                      "alphalib.Client.from_url", CLIENT, 1),
                  row(CM, 4, 1, "unverified", "method", (4,), "?.send", None,
                      2)])
    MO = "myapp/more.py"
    compare_rows("a walrus binds; `b = Base(); b.send()` resolved with the "
                 "old version; a changed constant used as a receiver is a "
                 "read of it; a local `def Client` hides the import for "
                 "the lines after it",
                 [as_row(h) for h in got.hits if h.file == MO],
                 [row(MO, 3, 10, "resolved", "call", CTOR, "alphalib.Client",
                      CLIENT, 1),
                  row(MO, 4, 5, "resolved", "method", (4,),
                      "alphalib.Client.send", BASE, 2),
                  row(MO, 6, 1, "resolved", "method", (4,),
                      "alphalib.Base.send", BASE, 2),
                  row(MO, 7, 5, "resolved", "attribute", (7,),
                      "alphalib.LIMIT"),
                  row(MO, 8, 5, "resolved", "attribute", (7,),
                      "alphalib.LIMIT")])
    check("the other files: factory's one call, the namespace package under "
          "lib/ (its factory's call, its handler's unverified .send on the "
          "factory's result), inner.py's call read under myapp/build, "
          "nothing from consumer.py, only.py, comment.py, build/ or the venv",
          sorted((h.file, h.line, h.change_ids, h.confidence)
                 for h in got.hits if h.file not in (SH, CR, GO, CM, MO)),
          [("lib/myapp2/factory.py", 5, CTOR, "resolved"),
           ("lib/myapp2/handlers.py", 4, (4,), "unverified"),
           ("myapp/build/inner.py", 2, (8,), "resolved"),
           ("myapp/factory.py", 5, CTOR, "resolved")])
    without = loc.locate(proj, changes() + extra_changes(), None)
    check("without the old version Base is not known to be a class: "
          "`b = Base(); b.send()` is unverified, the walrus's .send too "
          "(the bundle does not name Client.send), the constant's reads "
          "hold",
          [(h.line, h.confidence) for h in without.hits if h.file == MO],
          [(3, "resolved"), (4, "unverified"), (6, "unverified"),
           (7, "resolved"), (8, "resolved")])
    check("without the old version, the classmethod's explicit self is "
          "unknown and taken as explicit (R1 needs the old version anyway); "
          "the shadowing holds, and six more lines are unverified: the "
          "four Client.send calls the bundle does not name, the two "
          "Pool.acquire calls, Pool not being a known class, and "
          "Base().send",
          ([as_row(h)[11] for h in without.hits if h.file == CM],
           without.counts()[loc.UNVERIFIED],
           sorted(h.line for h in without.hits
                  if h.file == SH and h.confidence == loc.UNVERIFIED)
           == sorted([h.line for h in got.hits if h.file == SH
                      and h.confidence == loc.UNVERIFIED]
                     + [42, 43, 61, 73])),
          ([True, True, False], got.counts()[loc.UNVERIFIED] + 6, True))


# ------------------------------------------------------------ 6. report

def case_report(located: loc.Located) -> None:
    print("\n6. THE REPORT: PART 1 WHAT THE CODE USES, PART 2 THE REST")
    rels = [fb.Release("alphalib", "2.0.0", "2026-05-01", "analysed", 12)]
    ranked = sorted(changes(), key=lambda c: (-c.score, c.symbol, c.kind,
                                              c.sub_target))
    text = rep.report("alphalib", "1.0.0", "2.0.0", rels, ranked, located,
                      "2026-10-21T00:00:00Z", "/p/alpha_app",
                      ["old version alphalib 1.0.0 read from /s: its "
                       "classes' bases and signatures are known."])
    lines = text.splitlines()
    check("the header: the upgrade, the release crossed, the bundle's date",
          lines[:2],
          ["alphalib 1.0.0 -> 2.0.0: 1 release(s) crossed, 12 change(s); "
           "bundle built 2026-10-21T00:00:00Z",
           "  2.0.0          compared with the one before it: 12 change(s)"])
    check("the project line, the environment line and the counts",
          lines[3:6],
          ["project /p/alpha_app: 4 file(s) read, 3 mention the import "
           "root(s) alphalib",
           "  old version alphalib 1.0.0 read from /s: its classes' bases "
           "and signatures are known.",
           "  27 line(s) use 11 of the 12 change(s), 32 hit(s): 25 resolved, "
           "4 referenced, 3 unverified."])
    heads = [ln for ln in lines if re.match(r"^ {4}\d+  \S", ln)]
    check("PART 1's nine symbols in the site's order, numbered, with the "
          "label's count, the private mark and the line count",
          heads,
          ["    1  alphalib.core.Client   used by 9   (9 line(s))",
           "    2  alphalib.core.connect   used by 2   (1 line(s))",
           "    3  alphalib.core.helper   used by 1   (1 line(s))",
           "    4  alphalib.util.LIMIT   used by 3   (1 line(s))",
           "    5  alphalib.util.ping   used by 4   (1 line(s))",
           "    6  alphalib.core.Base.send   used by 5   (9 line(s))",
           "    7  alphalib.core.Client.__init__   used by 7   (10 line(s))",
           "    8  alphalib.core._internal   used by 0   private   "
           "(2 line(s))",
           "    9  alphalib.util.Pool.acquire   used by 0   (3 line(s))"])
    i = lines.index(heads[6])
    check("a function with three rows prints once, its rows under it, "
          "default before moved and db before port as the site orders them",
          lines[i + 1:i + 4],
          ["         PARAMETER_CHANGED_DEFAULT (retries)   3 -> 5",
           "         PARAMETER_MOVED (db)",
           "         PARAMETER_MOVED (port)"])

    def hit_line(where, confidence, use, source):
        # the report's columns, restated: 28, 11 and 9 wide, 7 in
        return f"       {where:<28} {confidence:<11} {use:<9} {source}"

    check("then its lines, file order, each with where, confidence, use and "
          "the source line",
          lines[i + 4:i + 7],
          [hit_line("app.py:12:10", "resolved", "call",
                    "client = Client(\"localhost\", 6380, 2)"),
           hit_line("app.py:18:14", "referenced", "argument",
                    "fn = partial(Client, \"h\")"),
           hit_line("app.py:24:9", "resolved", "call", "c = Client(host)")])
    check("a worker line under the same function, and the explicit-self "
          "call under Base.send",
          (hit_line("services/worker.py:21:17", "resolved", "call",
                    "worker_client = C(\"later\")") in text,
           hit_line("services/worker.py:10:5", "resolved", "method",
                    "Base.send(c, b\"p\", 3)") in text), (True, True))
    j = lines.index("PART 2  the upgrade's other 1 change(s), ranked, which "
                    "your code does not name:")
    check("PART 2: RATE alone, with its values",
          lines[j + 1],
          "       alphalib.util.RATE   ATTRIBUTE_CHANGED_VALUE   0.5 -> 0.25"
          "   used by 0")
    check("the notes, the once-sentence on what is invisible, no rewrites "
          "yet, and griffe credited",
          ("files the tool cannot see through, or did not read:" in text,
           "  services/legacy.py: line 2: `from alphalib.util import *`"
           in text,
           "getattr, strings and importlib are invisible to this tool. In "
           "the files that import the package, a method called on an "
           "object whose class the tool does not know is listed as "
           "unverified; an object made by another package's call or bound "
           "to a literal is not the package's and is not listed; a project "
           "module that re-exports the package's names is not followed."
           in text,
           "rewrites are not in this version" in text,
           lines[-1]),
          (True, True, True, True, "changes found with griffe."))
    nothing = loc.Located([], [], 2)
    text = rep.report("alphalib", "1.0.0", "2.0.0", rels, ranked, nothing,
                      "2026-10-21T00:00:00Z", "/p/none", ["x"])
    check("a project that names nothing: PART 1 says so, PART 2 holds all "
          "twelve",
          ("PART 1  no line of the project names a changed symbol" in text,
           "PART 2  the upgrade's other 12 change(s)" in text),
          (True, True))
    many = [change(100 + k, f"alphalib.m.f{k:04d}", "OBJECT_REMOVED")
            for k in range(250)]
    text = rep.report("alphalib", "1.0.0", "2.0.0", rels, many, nothing,
                      "d", "/p", [])
    check("PART 2 is cut at 200 with the count of the rest",
          ("       ... and 50 more" in text,
           text.count("OBJECT_REMOVED")), (True, 200))
    long = change(200, "alphalib.util.KEYS", "ATTRIBUTE_CHANGED_VALUE", "",
                  "", "(" + ", ".join(f"'k{i}'" for i in range(40)) + ")",
                  "()")
    line = rep.describe_change(long)
    check("a changed value longer than 72 characters is cut with an "
          "ellipsis in the one-line forms; a short one is whole",
          (len(line) < 200, line.count("\u2026"), "-> ()" in line,
           "   10 -> 20" in rep.describe_row(ranked[3])),
          (True, 1, True, True))
    text = rep.rank_report("alphalib", "1.0.0", "2.0.0", rels, ranked, "d",
                           "  --from defaulted to 1.0.0: why")
    check("--rank-only keeps the first week's list with its note, and says "
          "the project was not read",
          ("  --from defaulted to 1.0.0: why" in text,
           "    1  alphalib.core.Client (Base)   CLASS_REMOVED_BASE   "
           "used by 9" in text,
           "rank only: the project was not read (--rank-only)." in text),
          (True, True, True))


# ---------------------------------------------------------------- 7. cli

def write_bundle(path: pathlib.Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(fb.SCHEMA)
    con.executemany("INSERT INTO manifest VALUES (?, ?)",
                    [("schema_version", fb.SCHEMA_VERSION),
                     ("built_at", "2026-10-21T00:00:00Z")])
    con.execute("INSERT INTO package VALUES ('alphalib', 1)")
    con.executemany("INSERT INTO release VALUES (?, ?, ?, ?, ?)",
                    [("alphalib", "0.9.0", "2026-01-01", "no_baseline", 0),
                     ("alphalib", "1.0.0", "2026-02-01", "analysed_clean", 0),
                     ("alphalib", "2.0.0", "2026-05-01", "analysed", 12)])
    for c in changes():
        con.execute(
            "INSERT INTO change VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (c.id, c.package, c.version_from, c.version_to, c.symbol,
             ";".join(c.export_paths), c.kind, c.sub_target, c.message,
             c.old_value, c.new_value, int(c.is_private), c.module_depth,
             c.public_depth, c.inherited_by, c.user_count,
             c.alias_user_count, c.score))
    con.commit()
    con.close()


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def case_cli(tmp: pathlib.Path) -> None:
    print("\n7. THE COMMAND LINE: --from FROM THE ENVIRONMENT, EXIT CODES")
    bundle = tmp / "b.sqlite"
    write_bundle(bundle)
    base = ["alphalib", str(PROJECT), "--bundle", str(bundle)]
    code, out, err = run(base + ["--to", "2.0.0", "--env", str(SITE)])
    check("--from read from the environment's dist-info; the old version "
          "read; 27 lines, 32 hits; exit 1",
          (code, err,
           "alphalib 1.0.0 -> 2.0.0: 1 release(s) crossed, 12 change(s)"
           in out,
           f"old version alphalib 1.0.0 read from {SITE} (import roots: "
           "alphalib): its classes' bases and signatures are known." in out,
           "  27 line(s) use 11 of the 12 change(s), 32 hit(s): 25 resolved, "
           "4 referenced, 3 unverified." in out),
          (1, "", True, True, True))
    code, out, err = run(base + ["--from", "1.0.0", "--env", str(SITE)])
    check("--to defaults to the newest release; --from given agrees with the "
          "environment", (code, "alphalib 1.0.0 -> 2.0.0" in out,
                          "25 resolved" in out), (1, True, True))
    other = tmp / "other-site"
    (other / "alphalib-0.8.0.dist-info").mkdir(parents=True)
    (other / "alphalib-0.8.0.dist-info" / "METADATA").write_text(
        "Name: alphalib\nVersion: 0.8.0\n")
    code, out, err = run(base + ["--from", "1.0.0", "--to", "2.0.0",
                                 "--env", str(other)])
    check("the environment holds another version than --from: said, the old "
          "version not read, 11 unverified",
          (code, "the environment has alphalib 0.8.0, not 1.0.0, so the old "
                 "version's bases and signatures were not read" in out,
           "17 resolved, 4 referenced, 11 unverified" in out),
          (1, True, True))
    code, out, err = run(base + ["--to", "2.0.0", "--env", str(other)])
    check("--from not given and the environment's version is not in the "
          "bundle: refused with the bundle's newest",
          (code, "alphalib 0.8.0 is not a release the bundle holds (--from)"
           in err, "its newest alphalib is 2.0.0" in err), (2, True, True))
    empty = tmp / "empty-site"
    (empty / "zzz-1.0.dist-info").mkdir(parents=True)
    code, out, err = run(base + ["--to", "2.0.0", "--env", str(empty)])
    check("--from not given and the package not installed: exit 2 saying "
          "where it looked and to give --from",
          (code, "alphalib is not installed in" in err,
           "Give --from, or --env" in err), (2, True, True))
    code, out, err = run(base + ["--from", "1.0.0", "--to", "2.0.0",
                                 "--env", str(empty)])
    check("--from given, the package not installed: runs, says the old "
          "version was not read, exit 1",
          (code, "alphalib is not installed in" in out,
           "11 unverified" in out), (1, True, True))
    code, out, err = run(["alphalib", "--rank-only", "--to", "2.0.0",
                          "--bundle", str(bundle), "--env", str(empty)])
    check("--rank-only without an installed version: the release before "
          "--to, said in the header; the project unread; exit 0",
          (code, "  --from defaulted to 1.0.0, the release before 2.0.0: "
                 "alphalib is not installed in" in out,
           "rank only: the project was not read" in out,
           "PART 1" in out), (0, True, True, False))
    nothing = tmp / "quiet"
    nothing.mkdir()
    (nothing / "x.py").write_text("x = 1\n")
    code, out, err = run(["alphalib", str(nothing), "--from", "1.0.0",
                          "--to", "2.0.0", "--bundle", str(bundle),
                          "--env", str(SITE)])
    check("a project that names nothing: exit 0",
          (code, "PART 1  no line of the project names a changed symbol"
           in out), (0, True))
    code, out, err = run(["alphalib", str(tmp / "nowhere"), "--from",
                          "1.0.0", "--to", "2.0.0", "--bundle", str(bundle)])
    check("a project that is not a directory: exit 2",
          (code, "is not a directory" in err), (2, True))
    code, out, err = run(base + ["--from", "1.0.0", "--to", "2.0.0",
                                 "--env", str(tmp / "nowhere")])
    check("an --env that is not a directory: exit 2",
          (code, "is not a directory" in err), (2, True))
    code, out, err = run(base + ["--from", "0.5.0", "--to", "2.0.0",
                                 "--env", str(SITE)])
    check("a --from the bundle lacks: exit 2 with the bundle's newest",
          (code, "alphalib 0.5.0 is not a release the bundle holds" in err),
          (2, True))
    code, out, err = run(base + ["--from", "1.0", "--to", "2.0",
                                 "--env", str(SITE)])
    check("--from 1.0 and --to 2.0 are the bundle's 1.0.0 and 2.0.0",
          (code, "alphalib 1.0.0 -> 2.0.0: 1 release(s) crossed" in out),
          (1, True))
    saved = cli.locate
    cli.locate = lambda *a, **k: 1 / 0
    try:
        code, out, err = run(base + ["--from", "1.0.0", "--to", "2.0.0",
                                     "--env", str(SITE)])
    finally:
        cli.locate = saved
    check("an error the tool did not expect: one line on stderr, nothing on "
          "stdout, exit 2, no traceback",
          (code, out, err.startswith("the tool stopped: ZeroDivisionError"),
           "Traceback" in err), (2, "", True, False))
    code, out, err = run(["alphalib", "--rank-only", "--from", "1.0.0",
                          "--to", "2.0.0", "--bundle", str(bundle)])
    check("--rank-only with both versions: the ranked list, exit 0, the "
          "project unread",
          (code, "    1  alphalib.core.Client (Base)   CLASS_REMOVED_BASE"
           in out, "file(s) read" in out), (0, True, False))


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        case_env(tmp)
        case_oldapi()
        located = case_locate()
        case_locate_without_old()
        case_corners(tmp)
        case_review(tmp)
        case_report(located)
        case_cli(tmp)
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDo not ship locate until this passes.")
        sys.exit(1)
    print("All checks passed. Locate finds the fixture's lines, says how sure")
    print("it is of each, and the report ranks what the code uses first.")


if __name__ == "__main__":
    main()
