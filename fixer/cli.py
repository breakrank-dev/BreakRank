"""
The command line (NOTES §38.2, §38.5). In week 2 of the plan it does the
first of the three steps, rank: it names the releases an upgrade crosses,
with what the bundle says of each, and lists the upgrade's changes in the
site's order. Locate (the project's lines) and the two rewrite rules
follow in the weeks after, and until they are here the tool says so
rather than printing an empty section.

    python -m fixer <package> --from <version> --to <version> [project]
    python -m fixer <package> --to <version>           (--from: the newest
                                                        release before it
                                                        the bundle holds,
                                                        until the environment
                                                        is read, §38.2)
    python -m fixer --bundle path/to/breakrank.sqlite ...

Exit codes (§38.4): 0 nothing in the project touched, 1 lines flagged or
rewritten, 2 the tool could not run. With no locate step yet, 0 and 2.
"""

from __future__ import annotations

import argparse
import sys

from fixer import __version__
from fixer.bundle import Bundle, BundleError, Change, Release

SHOW = 60


def describe_change(c: Change) -> str:
    """One line per change for the report's ranked list."""
    what = c.symbol + (f" ({c.sub_target})" if c.sub_target else "")
    values = (f"   {c.old_value} -> {c.new_value}"
              if c.old_value is not None and c.new_value is not None else "")
    private = "   private" if c.is_private else ""
    return (f"{what}   {c.kind}{values}   used by {c.alias_user_count}"
            f"{private}")


def rank_report(package: str, frm: str, to: str, rels: list[Release],
                changes: list[Change], built_at: str) -> str:
    lines = [f"{package} {frm} -> {to}: {len(rels)} release(s) crossed, "
             f"{len(changes)} change(s); bundle built {built_at}"]
    for r in rels:
        lines.append(f"  {r.version:<14} {r.describe()}")
    not_compared = [r for r in rels if not r.compared]
    if not_compared:
        lines.append("  ** " + str(len(not_compared)) + " release(s) in the "
                     "range were not compared, so this list cannot be "
                     "complete for them.")
    lines.append("")
    if not changes:
        lines.append("no changes recorded in these releases.")
    else:
        lines.append("changes, in the site's order (shortest public path "
                     "first, then by name); \"used by\" counts the scanned "
                     "packages that\n  name the symbol by any of its paths "
                     "(NOTES §24):")
        for i, c in enumerate(changes[:SHOW], 1):
            lines.append(f"  {i:>3}  {describe_change(c)}")
        if len(changes) > SHOW:
            lines.append(f"       ... and {len(changes) - SHOW:,} more")
    lines.append("")
    lines.append("locate and the rewrites are not in this version: nothing "
                 "in the project was read (NOTES §38.3, §38.4 follow).")
    lines.append("changes found with griffe.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m fixer",
        description="BreakRank's fixer: rank, locate and fix the changes of "
                    "one upgrade of one package, offline.")
    ap.add_argument("package", nargs="?", help="the package upgraded")
    ap.add_argument("project", nargs="?", default=".",
                    help="the project directory (default: the current one)")
    ap.add_argument("--from", dest="frm", help="the version upgraded from")
    ap.add_argument("--to", help="the version upgraded to (default: the "
                                 "newest the bundle holds)")
    ap.add_argument("--bundle", help="a bundle file other than the tool's")
    ap.add_argument("--list", action="store_true",
                    help="list the packages the bundle holds and exit")
    ap.add_argument("--version", action="version",
                    version=f"breakrank-fix {__version__}")
    args = ap.parse_args(argv)

    try:
        bundle = Bundle(args.bundle) if args.bundle else Bundle.default()
    except BundleError as e:
        print(e, file=sys.stderr)
        return 2
    with bundle:
        if args.list:
            names = bundle.packages()
            print(f"{len(names)} package(s) in the bundle built "
                  f"{bundle.built_at()}:")
            print("\n".join(f"  {n}" for n in names))
            return 0
        if not args.package:
            ap.error("a package name is required (or --list)")
        try:
            rels = bundle.releases(args.package)
            to = args.to or rels[-1].version
            frm = args.frm
            if frm is None:
                # Until the environment is read (§38.2), the default is the
                # one-step upgrade: the newest release older than `to`.
                versions = [r.version for r in rels]
                if to not in versions:
                    raise BundleError(
                        f"{args.package} {to} is not a release the bundle "
                        f"holds (--to); its newest is {versions[-1]}.")
                older = versions[:versions.index(to)]
                if not older:
                    raise BundleError(
                        f"{args.package} {to} is the oldest release the "
                        "bundle holds; give --from.")
                frm = older[-1]
            crossed = bundle.upgrade(args.package, frm, to)
            changes = bundle.changes(args.package,
                                     [r.version for r in crossed])
        except BundleError as e:
            print(e, file=sys.stderr)
            return 2
        print(rank_report(args.package, frm, to, crossed, changes,
                          bundle.built_at()))
    return 0
