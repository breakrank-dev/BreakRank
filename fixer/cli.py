"""
The command line (NOTES §38.2, §38.3, §38.5). Two of the three steps are
here: rank, the upgrade's changes in the site's order with what the bundle
says of each release crossed, and locate, the project's lines that use
each changed name, with how sure the tool is of each. The rewrites (§38.4)
follow in the weeks after; until then every line is listed and none is
changed.

    python -m fixer <package> [project] --from <version> --to <version>
    python -m fixer <package> [project] --to <version>
        --from defaults to the version installed in the environment, read
        from its *.dist-info on disk (§38.2): the running interpreter's
        site-packages, or --env <venv root or site-packages directory>;
        when the package is not installed there the tool stops and says so.
    python -m fixer <package> --rank-only --from <version> --to <version>
        the ranked list alone; the project is not read. Without --from and
        without the package installed, this one defaults to the release
        before --to and says so in its header.
    python -m fixer --bundle path/to/breakrank.sqlite ...
    python -m fixer --list

The project defaults to the current directory. The old version's bases
and signatures are read with griffe from the environment when the
installed version is the one upgraded from; otherwise the report says
they were not, and methods called on instances are listed as unverified
where the bundle does not name the path.

Exit codes (§38.4): 0 nothing in the project names a changed symbol,
1 lines listed (later: flagged or rewritten), 2 the tool could not run,
including an error it did not expect (one line on stderr; the traceback
with BREAKRANK_DEBUG=1).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from fixer import __version__
from fixer.bundle import Bundle, BundleError, version_key
from fixer.env import EnvError, Installed, find_installed, site_packages
from fixer.locate import locate
from fixer.oldapi import OldApi
from fixer.report import rank_report, report


def read_environment(package: str, env: str | None
                     ) -> tuple[Installed | None, str]:
    """The installed distribution, or None and the sentence saying why."""
    try:
        return find_installed(package, env), ""
    except EnvError as e:
        return None, str(e)


def old_api(installed: Installed | None, frm: str, why: str
            ) -> tuple[OldApi | None, list[str]]:
    """griffe's reading of the old version, when the environment holds it,
    and the report's lines about it."""
    not_read = ("the old version's bases and signatures were not read; a "
                "method called on an instance is listed as unverified "
                "where the bundle does not name its path.")
    if installed is None:
        return None, [f"{why} So {not_read}"]
    if version_key(installed.version) != version_key(frm):
        return None, [f"the environment has {installed.distribution} "
                      f"{installed.version}, not {frm}, so {not_read}"]
    site = installed.dist_info.parent
    api = OldApi(installed.import_roots, site)
    roots = ", ".join(installed.import_roots) or "none found"
    if not api.loaded:
        reasons = "; ".join(f"{k}: {v}" for k, v in api.errors.items())
        return None, [f"griffe could not read {installed.distribution} "
                      f"{frm} from {site} (import roots: {roots}; "
                      f"{reasons or 'no import root'}), so {not_read}"]
    lines = [f"old version {installed.distribution} {frm} read from {site} "
             f"(import roots: {roots}): its classes' bases and signatures "
             "are known."]
    for root, err in api.errors.items():
        lines.append(f"  import root {root} could not be read: {err}")
    return api, lines


def spelled(rels, version: str) -> str:
    """The bundle's spelling of a version given another way (`2.0` for
    its `2.0.0`), or the version as given."""
    want = version_key(version)
    for r in rels:
        if version_key(r.version) == want:
            return r.version
    return version


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except (BundleError, EnvError) as e:
        print(e, file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - never a traceback, exit 2
        if os.environ.get("BREAKRANK_DEBUG"):
            raise
        print(f"the tool stopped: {type(e).__name__}: {e} (set "
              "BREAKRANK_DEBUG=1 for the traceback)", file=sys.stderr)
        return 2


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m fixer",
        description="BreakRank's fixer: rank, locate and fix the changes of "
                    "one upgrade of one package, offline.")
    ap.add_argument("package", nargs="?", help="the package upgraded")
    ap.add_argument("project", nargs="?", default=".",
                    help="the project directory (default: the current one)")
    ap.add_argument("--from", dest="frm",
                    help="the version upgraded from (default: the one "
                         "installed in the environment)")
    ap.add_argument("--to", help="the version upgraded to (default: the "
                                 "newest the bundle holds)")
    ap.add_argument("--env", help="the environment to read the installed "
                                  "version from: a venv root or a "
                                  "site-packages directory (default: this "
                                  "interpreter's)")
    ap.add_argument("--bundle", help="a bundle file other than the tool's")
    ap.add_argument("--rank-only", action="store_true",
                    help="list the upgrade's changes ranked and exit "
                         "without reading the project")
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
        if args.env is not None:
            try:
                site_packages(args.env)
            except EnvError as e:
                print(e, file=sys.stderr)
                return 2
        project = pathlib.Path(args.project)
        if not args.rank_only and not project.is_dir():
            print(f"{project} is not a directory (the project is the second "
                  "argument; the current directory when it is left out).",
                  file=sys.stderr)
            return 2

        installed, why = read_environment(args.package, args.env)
        note = ""
        try:
            rels = bundle.releases(args.package)
            to = spelled(rels, args.to) if args.to else rels[-1].version
            frm = spelled(rels, args.frm) if args.frm else None
            if frm is None and installed is not None:
                frm = spelled(rels, installed.version)
            if frm is None:
                if not args.rank_only:
                    raise BundleError(why)
                # --rank-only without an installed version: the one-step
                # upgrade, the newest release older than `to`, and say so.
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
                note = (f"  --from defaulted to {frm}, the release before "
                        f"{to}: {why}")
            crossed = bundle.upgrade(args.package, frm, to)
            changes = bundle.changes(args.package,
                                     [r.version for r in crossed])
        except BundleError as e:
            print(e, file=sys.stderr)
            return 2

        if args.rank_only:
            print(rank_report(args.package, frm, to, crossed, changes,
                              bundle.built_at(), note))
            return 0

        api, env_lines = old_api(installed, frm, why)
        located = locate(project, changes, api)
        print(report(args.package, frm, to, crossed, changes, located,
                     bundle.built_at(), str(project.resolve()), env_lines))
    return 1 if located.hits else 0
