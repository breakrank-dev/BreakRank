"""
Step 1 — get the list of the most-downloaded Python packages.

This is the top of the whole pipeline. Everything else works on the packages
this file hands back:

  * the top ~300 become "Track A" — we download every version and diff them
  * the top ~1,500 become "Track B" — we scan their source to see which
    symbols the ecosystem actually uses

Run it on its own to see the top 20:

    python ml/ingest/packages.py
"""

import json
import pathlib

import httpx

# The dataset is a nightly dump of PyPI download counts, published by hugovk.
# Two URLs for the same file. The first is the official home; the second is the
# raw file in the GitHub repo behind it. Some networks block one but not the
# other, so we try them in order.
TOP_URLS = [
    "https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json",
    "https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/top-pypi-packages.min.json",
]

# Where we keep a local copy, so you are not re-downloading 5 MB every run.
CACHE = pathlib.Path("data/top-pypi-packages.json")


def _fetch_raw(use_cache: bool = True) -> dict:
    """Download the raw JSON (or read the cached copy)."""
    if use_cache and CACHE.exists():
        return json.loads(CACHE.read_text())

    last_error = None
    for url in TOP_URLS:
        try:
            r = httpx.get(url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            data = r.json()
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(data))
            print(f"downloaded from {url}")
            return data
        except Exception as e:  # network error, 403, DNS failure...
            print(f"could not fetch {url}: {type(e).__name__}: {e}")
            last_error = e

    raise RuntimeError(
        "Could not download the package list from any URL. "
        "Check your internet connection, or try a phone hotspot."
    ) from last_error


def get_top_packages(limit: int = 2000, use_cache: bool = True) -> list[dict]:
    """
    Return the `limit` most-downloaded PyPI packages, rank 1 first.

    Each item looks like:
        {"name": "boto3", "downloads": 3206668324, "rank": 1}
    """
    data = _fetch_raw(use_cache=use_cache)

    # RULE FROM THE BOOK: always print the first item of any JSON you fetch
    # before writing code against it. Field names change without warning.
    # Verified 1 September 2026: rows are {"download_count": int, "project": str}
    rows = data["rows"]

    return [
        {"name": r["project"], "downloads": r["download_count"], "rank": i + 1}
        for i, r in enumerate(rows[:limit])
    ]


if __name__ == "__main__":
    packages = get_top_packages(2000)

    print(f"\nGot {len(packages)} packages. Top 20:\n")
    print(f"{'rank':>5}  {'package':<28} downloads")
    print("-" * 52)
    for p in packages[:20]:
        print(f"{p['rank']:>5}  {p['name']:<28} {p['downloads']:,}")

    print(f"\nTrack A (diff every version):  top 300, ending at "
          f"'{packages[299]['name']}'")
    print(f"Track B (scan for usage):      top 1500, ending at "
          f"'{packages[1499]['name']}'")
