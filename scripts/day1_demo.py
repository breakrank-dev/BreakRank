"""
Day 1 finish line — your first real rows of project data.

This runs the whole Track A pipeline end to end, small:

    top package list  ->  download 2 versions  ->  griffe diff  ->  CSV

on a handful of packages. It takes a few minutes and writes
data/day1_changes.csv.

    python scripts/day1_demo.py

When it finishes, open that CSV in Numbers or Excel. Those rows are real
breaking changes in real libraries that real people depend on. That file
is the first honest evidence that this project is going to work — and the
Sunday 13 September milestone is the same thing at 5,000 rows.
"""

import pathlib
import sys
import time

# Make "ml" importable when running this file directly from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from ml.ingest.api_extract import diff_versions
from ml.ingest.download import download_and_extract, list_releases

# Six well-known packages that ship sdists and have had real API churn.
# Small enough to finish over a coffee.
PACKAGES = ["click", "attrs", "jinja2", "werkzeug", "pygments", "packaging"]

OUT = pathlib.Path("data/day1_changes.csv")


def main() -> None:
    all_rows = []
    failures = []

    for package in PACKAGES:
        print(f"\n{'=' * 58}\n{package}\n{'=' * 58}")
        started = time.time()

        try:
            releases = list_releases(package, last_n=4)
        except Exception as e:
            print(f"  could not list releases: {type(e).__name__}: {e}")
            failures.append((package, "list_releases", str(e)))
            continue

        if len(releases) < 2:
            print("  fewer than 2 sdist releases — skipping")
            failures.append((package, "too_few_releases", ""))
            continue

        base = pathlib.Path(f"data/sdists/{package}")

        # Walk consecutive pairs: v1->v2, v2->v3, v3->v4
        for old_rel, new_rel in zip(releases, releases[1:]):
            label = f"{old_rel['version']} -> {new_rel['version']}"
            try:
                old_path = download_and_extract(old_rel["url"], base / old_rel["version"])
                new_path = download_and_extract(new_rel["url"], base / new_rel["version"])
                if old_path is None or new_path is None:
                    raise RuntimeError("could not work out the sdist layout")

                rows = diff_versions(package, old_path, new_path)
            except Exception as e:
                # THE RULE FROM THE BOOK: log every failure with a reason.
                # In week 2 you read these logs and find the one fix that
                # recovers fifty packages at once. That is real engineering,
                # and it is good interview material.
                print(f"  {label:<24} FAILED  {type(e).__name__}: {e}")
                failures.append((package, label, f"{type(e).__name__}: {e}"))
                continue

            for r in rows:
                r["version_from"] = old_rel["version"]
                r["version_to"] = new_rel["version"]
                r["released_at"] = new_rel["uploaded"][:10]

            all_rows.extend(rows)
            print(f"  {label:<24} {len(rows):>4} candidate changes")

        print(f"  ({time.time() - started:.0f}s)")

    if not all_rows:
        print("\nNo rows produced at all. Check your internet and rerun.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df = df[
        [
            "package", "version_from", "version_to", "symbol", "kind",
            "is_private", "module_depth", "is_top_level", "name_length",
            "released_at", "explanation",
        ]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 58)
    print(f"  {len(df):,} candidate breaking changes  ->  {OUT}")
    print("=" * 58)

    print(f"\nAcross {df['package'].nunique()} packages and "
          f"{len(df.groupby(['package', 'version_from', 'version_to']))} version pairs.")

    print("\nKinds of change found:")
    counts = df["kind"].value_counts()
    for kind, n in counts.items():
        print(f"  {n:>5}  ({n / len(df):>4.0%})  {kind}")

    # ---- the point of the whole project, visible on day 1 -----------------
    top_kind, top_n = counts.index[0], counts.iloc[0]
    print(f"\nLook at that top row. {top_n} of {len(df)} rows ({top_n / len(df):.0%}) are")
    print(f"{top_kind}. Almost none of those will break anybody.")
    print("Meanwhile OBJECT_REMOVED — a function that is simply gone — is a")
    print("handful of rows buried underneath.")
    print()
    print("That is the problem, on your own screen, on day 1: the signal is")
    print("real but it is drowning. Sorting this list correctly is BreakRank.")

    private_share = df["is_private"].mean()
    print(f"\n({private_share:.0%} of these are private symbols — names starting with _.")
    print(" One free column, and it already removes rows nobody could ever call.")
    print(" You will add a dozen more features like it in week 4.)")

    if failures:
        print(f"\n{len(failures)} failure(s) logged — this is normal and expected:")
        for pkg, where, why in failures[:8]:
            print(f"  {pkg:<12} {where:<24} {why[:60]}")
        print("Do not fix these today. Week 2 is when you read the log and find")
        print("the one fix that recovers fifty packages at once.")

    size_mb = sum(f.stat().st_size for f in pathlib.Path("data/sdists").rglob("*") if f.is_file()) / 1e6
    print(f"\nDownloaded source now on disk: {size_mb:,.0f} MB in data/sdists/")
    print("data/ is gitignored, so none of this goes near your repo. Delete the")
    print("folder any time you need the space — it re-downloads.")

    print("\nOpen the CSV and scroll it for five minutes before you stop for the day.")


if __name__ == "__main__":
    main()
