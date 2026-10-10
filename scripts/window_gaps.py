"""
How many releases did we step over, and why?

    python scripts/window_gaps.py
    python scripts/window_gaps.py --workers 12 --out data/window_gaps.csv

Varad asked for this on 15 Sep, and he was right to: "wheel-only releases
are probably rare" does not survive a review. "0.x% of releases in our
windows, here is the table" does.

THE OFF-BY-ONE THIS MEASURES.

griffe reads source. A release that ships only a pre-built wheel has no
source to read, so the pipeline skips it — and a break introduced *in*
that release gets attributed to the next release that did publish an
sdist. The chain still looks unbroken. Nothing in failures.csv records
it, because nothing failed.

Same for a yanked release (correctly skipped: nobody upgraded *through*
one) and, until 14 Sep, for pre-releases — which turned out not to be
rare at all and cost us eight packages (NOTES §10.6).

SO: for every package, take the version range we actually analysed, ask
PyPI for everything published inside that range, and account for each
release that is not in changes.csv. Each is one of five things, and only
one of them is a defect:

  yanked          correct. Withdrawn by its authors.
  pre-release     correct where a package has real releases; §10.6 covers
                  the case where it does not.
  no sdist        THE OFF-BY-ONE. Source was never published, so any
                  break in it is attributed to a later version.
  left no row     analysed, nothing found, no trace written. See below.
  unexplained     should be zero. If it is not, the window logic and this
                  script disagree and one of them is wrong.

WHAT THE FIFTH CATEGORY TURNED OUT TO BE. It was written as a paranoid
consistency check and expected to print zero. It printed 27, and the
cause is the quiet-release gap arriving from an unfamiliar direction.

`used` is derived from changes.csv, and a version pair that produces no
breaking changes writes no rows — so a release whose BOTH adjacent pairs
were clean disappears from that file entirely. It was downloaded, parsed
and diffed, and nothing in our data records that it exists.

The inference is safe in the other direction too: a release strictly
inside [oldest used, newest used] that is eligible MUST have been among
the last six eligible releases, so it must have been analysed. If it had
failed it would be in failures.csv. Neither holds, so it was analysed and
was clean.

NOTE THE COUNT IS A LOWER BOUND ON THE REAL THING. It counts invisible
RELEASES, which need two clean pairs each. Clean PAIRS are strictly more
numerous and cannot be counted from outside at all — that needs the
ingest to record every release it considered, which is the work behind
Varad's analysis_status = 'analysed_clean'.

Read-only: one GET per package, nothing written unless --out is passed.
"""

import argparse
import collections
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.ingest.download import get_json  # noqa: E402

CHANGES = pathlib.Path("data") / "changes.csv"


def parse(v: str) -> Version | None:
    try:
        return Version(v)
    except InvalidVersion:
        return None


def audit(package: str, used: set[str],
          failed_packages: frozenset[str] = frozenset()) -> list[dict]:
    """Account for every release between the oldest and newest we used."""
    vs = [v for v in (parse(u) for u in used) if v]
    if len(vs) < 2:
        return []
    lo, hi = min(vs), max(vs)

    data = get_json(f"https://pypi.org/pypi/{package}/json")
    out = []
    for version, files in data["releases"].items():
        v = parse(version)
        if v is None or not (lo <= v <= hi):
            continue
        if version in used:
            out.append({"package": package, "version": version,
                        "status": "used"})
            continue

        # ORDER MATTERS HERE, and it is the same order list_releases
        # applies. A yanked pre-release with no sdist is reported once,
        # under the first reason that would have excluded it — otherwise
        # the columns sum to more than the rows and the table lies.
        if v.is_devrelease:
            status = "dev release"
        elif not files:
            status = "no files at all"
        elif all(f.get("yanked") for f in files):
            status = "yanked"
        elif v.is_prerelease:
            status = "pre-release"
        elif not any(f["packagetype"] == "sdist" and not f.get("yanked")
                     for f in files):
            status = "no sdist"
        elif package in failed_packages:
            # Eligible, inside the window, and this package logged a
            # failure — so it was probably a failed pair rather than a
            # clean one. Named separately because the two mean opposite
            # things: one is missing data, the other is good news.
            status = "analysed, package had a failure"
        else:
            status = "left no row (analysed, nothing found)"
        out.append({"package": package, "version": version, "status": status})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Account for releases inside our analysed windows.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="optional CSV of every release and its status")
    args = ap.parse_args()

    if not CHANGES.exists():
        sys.exit(f"{CHANGES} not found.")
    ch = pd.read_csv(CHANGES)
    # Built by hand rather than with groupby().apply(), which needs the
    # include_groups argument on pandas >= 2.2 and warns without it on
    # older ones. A dict comprehension has no version opinion.
    used: dict[str, set[str]] = {}
    for pkg, a, b in zip(ch["package"], ch["version_from"], ch["version_to"]):
        used.setdefault(pkg, set()).update((str(a), str(b)))
    fp = pathlib.Path("data") / "failures.csv"
    failed_packages = frozenset(
        pd.read_csv(fp)["package"]) if fp.exists() else frozenset()

    print(f"\n{len(used)} packages analysed. Asking PyPI what else was "
          f"published inside\ntheir version windows — one request each, "
          f"{args.workers} at a time.\n")

    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(audit, p, u, failed_packages): p
                for p, u in used.items()}
        for i, f in enumerate(as_completed(futs), 1):
            pkg = futs[f]
            try:
                rows.extend(f.result())
            except Exception as e:
                failed.append((pkg, type(e).__name__))
            if i % 50 == 0:
                print(f"  {i}/{len(futs)}")

    if failed:
        print(f"\n{len(failed)} packages could not be checked "
              f"(counted nowhere below): {[p for p, _ in failed][:8]}")

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("nothing to report — no package had two usable versions.")

    counts = collections.Counter(df["status"])
    total = len(df)
    skipped = total - counts["used"]

    print("\n" + "=" * 64)
    print("  EVERY RELEASE INSIDE OUR ANALYSED WINDOWS")
    print("=" * 64)
    print(f"  releases in range   {total:>6,}")
    quiet_n = counts.get("left no row (analysed, nothing found)", 0)
    print(f"  in changes.csv      {counts['used']:>6,}  "
          f"({counts['used']/total:.1%})")
    # "not in changes.csv" rather than "skipped": some of these WERE
    # analysed and simply produced no row, which is the opposite of
    # skipped and the whole point of the fifth category.
    print(f"  not in changes.csv  {skipped:>6,}  ({skipped/total:.1%})"
          + (f"   of which {quiet_n} were analysed" if quiet_n else "") + "\n")

    for status in ("no sdist", "yanked", "pre-release", "dev release",
                   "no files at all", "left no row (analysed, nothing found)",
                   "analysed, package had a failure", "unexplained"):
        n = counts.get(status, 0)
        if not n:
            continue
        note = {
            "no sdist": "<- THE OFF-BY-ONE. Source never published.",
            "left no row (analysed, nothing found)": "<- QUIET RELEASES",
            "unexplained": "<- SHOULD BE ZERO. Investigate.",
        }.get(status, "")
        print(f"  {status:<38} {n:>6,}  ({n/total:>5.2%})  {note}")

    ns = counts.get("no sdist", 0)
    print("\n" + "-" * 64)
    print(f"  THE NUMBER FOR THE WRITE-UP: {ns/total:.2%} of releases inside")
    print("  our windows shipped no source. A breaking change introduced in")
    print("  one of those is attributed to the next release that did publish")
    print("  source. That is a known, measured off-by-one — not an estimate.")

    quiet = counts.get("left no row (analysed, nothing found)", 0)
    if quiet:
        print(f"\n  {quiet} releases ({quiet/total:.2%}) were analysed, broke")
        print("  nothing, and left NO TRACE in changes.csv — a pair with zero")
        print("  changes writes zero rows, so a release whose both adjacent")
        print("  pairs were clean vanishes. 'We checked and it is safe' is")
        print("  currently indistinguishable from 'we never looked'.")
        print("  A LOWER BOUND: invisible releases need two clean pairs each,")
        print("  so clean PAIRS are more numerous and cannot be counted from")
        print("  outside. That needs the ingest to record what it considered.")

    if counts.get("unexplained"):
        print("\n  ** 'unexplained' is not zero. Either list_releases and this")
        print("  ** script disagree about the rules, or a release is being")
        print("  ** dropped for a reason neither of us has written down.")
        for _, r in df[df["status"] == "unexplained"].head(8).iterrows():
            print(f"       {r['package']} {r['version']}")

    worst = (df[df["status"] == "no sdist"].groupby("package").size()
             .sort_values(ascending=False))
    if len(worst):
        print("\n  packages losing the most releases to wheel-only publishing:")
        for pkg, n in worst.head(8).items():
            inwin = int((df["package"] == pkg).sum())
            print(f"    {pkg:<26} {n} of {inwin} releases in window")

    if args.out:
        df.sort_values(["package", "version"]).to_csv(args.out, index=False)
        print(f"\n  full table -> {args.out}")
        print("  This is also the source for analysis_status = 'no_source'")
        print("  and 'yanked' in Varad's release table.")


if __name__ == "__main__":
    main()
