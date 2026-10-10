"""
Do the two HISTORY features say what they claim, and only that?

    python scripts/test_history.py

No network, no sdists, no changes.csv. It writes tiny three-version
packages into a temp directory, runs the real diff_series(), and checks
the two columns. Under a second, so it can be run before every ingest —
which matters more here than usual, because a re-ingest is hours long and
both of these columns are written ONCE, at extract time, and cannot be
recomputed afterwards from what lands in changes.csv.

WHY THIS FILE EXISTS.

`was_deprecated_before` is the feature the project book expects to be the
strongest in the set, and three separate measurements say the obvious
implementation of it returns False on every row:

  1. griffe.deprecated is None for a DOCSTRING marker. Fair enough —
     prose is not metadata.
  2. griffe.deprecated is ALSO None for an actual @deprecated DECORATOR,
     on griffe 2.2.0. That one is not obvious, and a feature that reads
     only that flag would report zero gain and look like the finding
     "deprecation does not predict breakage" rather than a broken reader.
  3. Decorators come back as STRINGS — decorators=["deprecated('use
     plain instead')"] — so they have to be matched as text.

And one structural trap on top: the paths users actually write are
usually re-exported ALIASES (pkg.thing, not pkg.core.thing), and reading
.docstring on an alias raises AliasResolutionError — NOTES §9.1. Skipping
aliases is therefore mandatory, and skipping them naively lands the
signal on exactly the rows nobody imports. deprecated_paths() collects
definition paths per version so an alias row can be answered through
target_path without ever resolving it. Case 2 below is that gap; it
failed before the fix and is the reason the function exists.

`prior_breaks_in_module` has one way to be wrong that no amount of
staring at the output would catch: counting the row's own pair, or a
later one. A feature that can see the future scores beautifully and
predicts nothing. Case 3 pins the count to what came strictly BEFORE.
"""

import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.ingest.api_extract import diff_series  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def write(root: pathlib.Path, core: str, init: str) -> pathlib.Path:
    """One version of a package: pkg/core.py plus a re-exporting __init__."""
    pkg = root / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    (pkg / "core.py").write_text(core)
    (pkg / "__init__.py").write_text(init)
    return root


def series(tmp: pathlib.Path, versions: list[tuple[str, str, str]]):
    """Run the real diff_series over N versions, oldest first.

    Returns a flat list of rows with version_from/version_to attached, the
    way run_ingest attaches them — diff_collections never sees a version
    number, it only ever sees two loaded collections, so the pairing has
    to be done by whoever consumes the generator.
    """
    on_disk = []
    for i, (v, core, init) in enumerate(versions):
        root = tmp / f"v{i}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        write(root, core, init)
        on_disk.append((v, root))

    rows = []
    for vf, vt, produced in diff_series("pkg", on_disk, ["pkg"]):
        if isinstance(produced, Exception):
            raise produced
        for r in produced:
            rows.append({**r, "version_from": vf, "version_to": vt})
    return rows


# ------------------------------------------------------------------ cases

DEPRECATED_CORE = (
    # A decorator FACTORY whose own name contains the stem the feature
    # matches on, carrying no marker of its own. If was_deprecated() ever
    # drifts into matching names instead of decorators and docstrings,
    # this row turns True and the case below catches it.
    "def deprecated(msg):\n"
    "    def wrap(fn):\n"
    "        return fn\n"
    "    return wrap\n"
    "\n\n"
    "@deprecated('use plain instead')\n"
    "def a():\n"
    "    return 1\n"
    "\n\n"
    "def b():\n"
    "    '''Old helper.\n"
    "\n"
    "    .. deprecated:: 1.0\n"
    "       Use plain() instead.\n"
    "    '''\n"
    "    return 2\n"
    "\n\n"
    "def d():\n"
    "    '''A perfectly ordinary function.'''\n"
    "    return 4\n"
    "\n\n"
    "def c():\n"
    "    return 3\n"
)
# __all__ IS NOT DECORATION HERE. griffe reports the re-exported alias as
# its own breakage only when the package declares the name in __all__ —
# measured: a plain `from .core import a` yields ['pkg.core.a'] alone,
# while adding __all__ = ['a'] yields ['pkg.a', 'pkg.core.a']. Since the
# alias rows are the entire point of case 2, a fixture without __all__
# would produce no alias rows and the case would pass by having nothing
# to check.
REEXPORT = ("from .core import a, b, c, d  # noqa: F401\n"
            "__all__ = ['a', 'b', 'c', 'd']\n")
SURVIVOR_CORE = "def c():\n    return 3\n"
SURVIVOR_INIT = ("from .core import c  # noqa: F401\n"
                 "__all__ = ['c']\n")


def case_marker_forms(tmp: pathlib.Path) -> None:
    print("\n1. BOTH MARKER FORMS — decorator and docstring")
    rows = series(tmp / "markers", [
        ("1.0", DEPRECATED_CORE, REEXPORT),
        ("2.0", SURVIVOR_CORE, SURVIVOR_INIT),
    ])
    got = {r["symbol"]: r["was_deprecated_before"] for r in rows}

    check("decorator marker is read", got.get("pkg.core.a"), True)
    check("docstring marker is read", got.get("pkg.core.b"), True)
    # The controls. Without them this case passes just as well for a
    # function that returns True unconditionally, which is the failure
    # mode a "strongest feature" is most likely to have and least likely
    # to be questioned about.
    check("an UNMARKED symbol stays False", got.get("pkg.core.d"), False)
    check("the NAME 'deprecated' is not a marker",
          got.get("pkg.core.deprecated"), False)


def case_alias_rows(tmp: pathlib.Path) -> None:
    print("\n2. THE RE-EXPORTED PATHS — what users actually import")
    rows = series(tmp / "alias", [
        ("1.0", DEPRECATED_CORE, REEXPORT),
        ("2.0", SURVIVOR_CORE, SURVIVOR_INIT),
    ])
    got = {r["symbol"]: r["was_deprecated_before"] for r in rows}

    # THIS IS THE CASE THAT FAILED FIRST. Aliases are skipped on purpose —
    # touching .docstring on one raises AliasResolutionError — and the
    # first version of was_deprecated() therefore returned False for
    # pkg.a and pkg.b while returning True for pkg.core.a and pkg.core.b.
    # Correct by the letter, and it put the signal on the definition paths
    # and withheld it from the import paths, which are the rows the whole
    # feature is supposed to be about.
    check("alias row inherits the decorator marker", got.get("pkg.a"), True)
    check("alias row inherits the docstring marker", got.get("pkg.b"), True)
    check("unmarked alias stays False", got.get("pkg.d"), False)
    check("no row was lost to the lookup", len(rows), len(got))


NOISE_CORE = '''
class filterwarnings:
    def __init__(self, *a):
        pass

    def __call__(self, fn):
        return fn


def deprecate_kwarg(old, new):
    def wrap(fn):
        return fn
    return wrap


def deprecated(msg):
    def wrap(fn):
        return fn
    return wrap


class WarningClass(Exception):
    """Issued for usage of deprecated APIs."""


@filterwarnings("ignore::DeprecationWarning")
def suppressor():
    """A test helper."""
    return 1


@deprecate_kwarg("old_name", "new_name")
def renamed_kwarg():
    """Still perfectly supported, just called differently."""
    return 2


def documents_a_param(flag=False):
    """Do a thing.

    :param flag: If True, issues a message indicating that the command is
        deprecated and highlights its deprecation in --help.
    """
    return 3


def mode_is_going(mode="full"):
    """Compute a thing.

    The 'economic' mode is deprecated.  Use 'reduced' instead.
    """
    return 4


@deprecated("use plain instead")
def real_positive():
    """The control: one genuinely deprecated symbol in the same file."""
    return 5


def survivor():
    return 6
'''


def case_talks_about_vs_is(tmp: pathlib.Path) -> None:
    print("\n4. TALKING ABOUT DEPRECATION IS NOT BEING DEPRECATED")
    # Every shape here was found by surveying real packages, not invented.
    # The bare-stem version of was_deprecated() returned True for all five
    # of the noise rows below, which would have made the book's "single
    # strongest feature" about half noise on the docstring path and about
    # 60% noise on the decorator path. NOTES §16.6.
    # Explicit imports, never `from .core import *` — griffe records a
    # star-import as a single pseudo-member called "pkg/core/*" and the
    # pair then produces exactly one row for the wildcard instead of one
    # per symbol. The first version of this fixture did that and the case
    # "passed" its way to six Nones.
    noise_init = ("from .core import (WarningClass, documents_a_param,\n"
                  "                   mode_is_going, real_positive,\n"
                  "                   renamed_kwarg, suppressor, survivor)\n")
    rows = series(tmp / "noise", [
        ("1.0", NOISE_CORE, noise_init),
        ("2.0", "def survivor():\n    return 6\n",
         "from .core import survivor  # noqa: F401\n"),
    ])
    got = {r["symbol"]: r["was_deprecated_before"] for r in rows}

    check("the warning CLASS is not deprecated (sqlalchemy.exc shape)",
          got.get("pkg.core.WarningClass"), False)
    check("@filterwarnings('ignore::DeprecationWarning') is not a marker",
          got.get("pkg.core.suppressor"), False)
    check("@deprecate_kwarg deprecates an ARGUMENT, not the function",
          got.get("pkg.core.renamed_kwarg"), False)
    check("documenting a parameter named 'deprecated' is not a marker",
          got.get("pkg.core.documents_a_param"), False)
    check("'the economic mode is deprecated' is about a MODE (numpy shape)",
          got.get("pkg.core.mode_is_going"), False)
    # Without this line the whole case passes for a function that returns
    # False unconditionally — which is exactly the failure a tightening
    # pass is most likely to introduce and least likely to notice.
    check("and a REAL decorator marker still reads True",
          got.get("pkg.core.real_positive"), True)


def case_no_lookahead(tmp: pathlib.Path) -> None:
    print("\n3. prior_breaks_in_module COUNTS ONLY THE PAST")
    # Three versions, one removal from pkg.core each time. The first pair
    # has nothing behind it; each later pair has exactly the pairs before
    # it, and never its own row.
    v1 = "def x():\n    return 1\n\n\ndef y():\n    return 2\n\n\ndef z():\n    return 3\n"
    v2 = "def y():\n    return 2\n\n\ndef z():\n    return 3\n"
    v3 = "def z():\n    return 3\n"
    rows = series(tmp / "history", [
        ("1.0", v1, "from .core import x, y, z  # noqa: F401\n"),
        ("2.0", v2, "from .core import y, z  # noqa: F401\n"),
        ("3.0", v3, "from .core import z  # noqa: F401\n"),
    ])

    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["version_from"], r["version_to"]), []).append(r)

    first = [r for r in by_pair.get(("1.0", "2.0"), [])
             if r["symbol"] == "pkg.core.x"]
    second = [r for r in by_pair.get(("2.0", "3.0"), [])
              if r["symbol"] == "pkg.core.y"]

    check("first pair sees no history", first[0]["prior_breaks_in_module"]
          if first else None, 0)
    # 1.0 -> 2.0 removed x under two paths (pkg.x and pkg.core.x), so the
    # module pkg.core has 2 breakages behind it by the time 2.0 -> 3.0 is
    # diffed. The number to defend is "strictly earlier", not a particular
    # integer — so assert the relation, not a magic constant.
    prior = second[0]["prior_breaks_in_module"] if second else None
    check("second pair sees the first pair", prior, len(by_pair[("1.0", "2.0")]))
    check("and does NOT count itself or later pairs",
          prior is not None and prior < len(rows), True)

    # Different module, same chain: history is per-module, not per-package.
    check("every row of one pair shares one count",
          len({r["prior_breaks_in_module"] for r in by_pair[("1.0", "2.0")]
               if r["symbol"].startswith("pkg.core.")}), 1)


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-history-"))
    try:
        case_marker_forms(tmp)
        case_alias_rows(tmp)
        case_no_lookahead(tmp)
        case_talks_about_vs_is(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Deprecation is read from decorators AND")
    print("docstrings, reaches the re-exported paths users import, and")
    print("module history never counts the present or the future.")


if __name__ == "__main__":
    main()
