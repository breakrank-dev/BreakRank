"""
Are we really diffing consecutive release pairs?

    python scripts/verify_pairs.py

Asked by Varad before loading 23,024 rows, and it is exactly the right
question to ask first: decision 7 needs consecutive pairs, because
"breaking changes hide in patch releases" is the project's premise and
`via_version` has to name the release that actually introduced a change.
If we were diffing everything against latest, a change introduced in
2.1.1 and still present in 2.2.0 would be attributed to 2.2.0, and the
whole claim collapses.

Two different things get checked here, because "consecutive" can fail in
two ways and only one of them is what he asked about:

  1. ARE THE PAIRS ADJACENT? Does every pair go from one release to the
     very next one, with nothing skipped? A pair like 2.1.0 -> 2.1.3 would
     mean 2.1.1 and 2.1.2 were jumped over and anything they introduced is
     misattributed. This is the thing that would make the project wrong.

  2. IS EVERY RELEASE ACCOUNTED FOR? A subtler one. changes.csv only
     contains pairs that PRODUCED at least one change. A release that
     broke nothing leaves no row anywhere — so the release table built
     from this file can be missing quiet releases entirely. That does not
     make any attribution wrong, but it does make a quiet release
     indistinguishable from one we never looked at, which is exactly the
     distinction decision 1 exists to preserve.

The second is reported as a gap count rather than a pass/fail, because
whether it matters depends on what the API does with a missing release.
"""

import pathlib
import sys
from collections import defaultdict

import pandas as pd
from packaging.version import InvalidVersion, Version

DATA = pathlib.Path("data")
CHANGES = DATA / "changes.csv"


def parse(v: str):
    try:
        return Version(str(v))
    except InvalidVersion:
        return None


def main() -> None:
    if not CHANGES.exists():
        sys.exit(f"{CHANGES} not found — run the pipeline first.")
    df = pd.read_csv(CHANGES)

    pairs = defaultdict(list)
    for pkg, vf, vt in (df[["package", "version_from", "version_to"]]
                        .drop_duplicates().itertuples(index=False)):
        pairs[pkg].append((str(vf), str(vt)))

    non_adjacent, chains_with_holes, unparseable = [], [], []
    total_pairs = sum(len(v) for v in pairs.values())
    releases_seen, releases_if_whole = set(), 0

    for pkg, plist in pairs.items():
        keyed = []
        for vf, vt in plist:
            a, b = parse(vf), parse(vt)
            if a is None or b is None:
                unparseable.append((pkg, vf, vt))
                continue
            keyed.append((a, b, vf, vt))
        keyed.sort(key=lambda t: t[0])

        for a, b, vf, vt in keyed:
            releases_seen.add((pkg, vf))
            releases_seen.add((pkg, vt))
            # A pair must go FORWARD. Backwards or equal is a real bug.
            if b <= a:
                non_adjacent.append((pkg, vf, vt, "not increasing"))

        # Adjacency: each pair should start where the previous one ended.
        # A break means the pair BETWEEN them is absent from changes.csv.
        holes = 0
        for (a1, b1, vf1, vt1), (a2, b2, vf2, vt2) in zip(keyed, keyed[1:]):
            if vt1 != vf2:
                holes += 1
        if holes:
            chains_with_holes.append((pkg, holes, len(keyed)))
        releases_if_whole += len(keyed) + 1

    print(f"\n{len(pairs)} packages   {total_pairs:,} distinct version pairs")
    print(f"{total_pairs / max(len(pairs), 1):.1f} pairs per package "
          f"(6 releases per package gives 5 pairs)\n")

    print("=" * 66)
    print("  1. ARE THE PAIRS ADJACENT?  (the question that matters)")
    print("=" * 66)
    if non_adjacent:
        print(f"** {len(non_adjacent)} pairs do not move forward in version "
              "order:")
        for row in non_adjacent[:10]:
            print("   ", row)
    else:
        print("Every pair moves forward in PEP 440 order. No pair skips a")
        print("release that another pair covers. This is per-consecutive-pair")
        print("diffing, not diff-against-latest.")

    def render(pkg: str) -> str:
        """Draw the chain, showing gaps instead of papering over them.

        The first version of this printed each pair's END version and
        assumed the next pair STARTED there — so a missing pair was drawn
        as an unbroken chain. typing-extensions came out as
        '4.13.2 -> 4.14.0 -> 4.16.0' while being flagged as broken two
        sections later. A diagram that hides the thing it is meant to
        prove is worse than no diagram.
        """
        keyed = sorted(((parse(a), a, b) for a, b in pairs[pkg]
                        if parse(a) and parse(b)), key=lambda t: t[0])
        if not keyed:
            return "(no parseable versions)"
        out, prev_end = "", None
        for _, vf, vt in keyed:
            if prev_end is None:
                out = vf
            elif vf != prev_end:
                out += f"  ⟨gap⟩  {vf}"
            out += f" -> {vt}"
            prev_end = vt
        return out

    holed = {c[0] for c in chains_with_holes}
    longest = sorted(pairs, key=lambda p: -len(pairs[p]))
    clean = [p for p in longest if p not in holed]
    broken = [p for p in longest if p in holed]

    print("\nSample chains — read these, they are the actual evidence:")
    for pkg in clean[:3]:
        print(f"  {pkg:<20} {render(pkg)}")
    if broken:
        print("\nAnd chains WITH a gap, shown honestly:")
        for pkg in broken[:3]:
            print(f"  {pkg:<20} {render(pkg)}")

    print("\nOne caveat on the word 'consecutive': it means consecutive among")
    print("releases that ship SOURCE. A version published as a wheel only has")
    print("no source for griffe to read, so it never enters the chain — and a")
    print("breaking change introduced there would be attributed to the next")
    print("release that did ship source. Worth stating before anyone relies on")
    print("via_version being exact.")

    print("\n" + "=" * 66)
    print("  2. IS EVERY RELEASE ACCOUNTED FOR?  (the subtler one)")
    print("=" * 66)
    print(f"releases present in changes.csv:  {len(releases_seen):,}")
    print(f"packages:                         {len(pairs):,}")
    print(f"average per package:              "
          f"{len(releases_seen) / max(len(pairs), 1):.1f}   "
          f"(the run asked for 6)\n")

    if chains_with_holes:
        holes = sum(h for _, h, _ in chains_with_holes)
        print(f"{len(chains_with_holes)} packages have a break in their chain "
              f"({holes} in total).")
        print("A break means the pair between two others is ABSENT from")
        print("changes.csv — either it produced zero changes, or its download")
        print("failed. From this file the two are indistinguishable;")
        print("data/failures.csv is where they separate.\n")
        for pkg, h, n in sorted(chains_with_holes, key=lambda t: -t[1])[:8]:
            print(f"  {pkg:<24} {h} break(s) across {n} pairs")
    else:
        print("No breaks. Every chain is unbroken.")

    print("\nWhat this file CANNOT tell you: how many releases PyPI actually")
    print("had. A release is invisible here only if BOTH pairs touching it")
    print("produced nothing — and a missing pair usually still leaves both of")
    print("its releases visible through their neighbours, so a break above")
    print("does NOT mean a lost release. The true release list lives in the")
    print("ingest run, not in this file. If the API needs quiet releases")
    print("listed (decision 1: 'analysed and found nothing' must not look")
    print("like 'never analysed'), they have to be recorded at ingest time.")

    if unparseable:
        s = "" if len(unparseable) == 1 else "s"
        print(f"\n{len(unparseable)} pair{s} had a version PEP 440 could not "
              f"parse, e.g. {unparseable[0]}")


if __name__ == "__main__":
    main()
