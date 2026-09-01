"""
Step 3 — read a package's API with griffe, and diff two versions.

This is the step that produces your CANDIDATE CHANGES: every difference
griffe considers potentially breaking between version A and version B.
There will be roughly 200 per major release and about 5 will actually
matter — ranking them is the entire project.

Nothing here is machine learning yet. This is the machine that makes the data.
"""

import logging
import re

import griffe

# griffe prints a lot of "could not resolve alias" warnings on real packages.
# They are harmless — it just means one module referenced something it could
# not follow. Silence them so your own output stays readable.
logging.getLogger("griffe").setLevel(logging.ERROR)

# b.explain() returns text with terminal colour codes baked in. Those look
# like garbage in a CSV, so strip them.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_api(package_name: str, search_path) -> griffe.Module | None:
    """
    Read one version of a package into a griffe object tree.

    `search_path` is the folder Python would import FROM — the thing
    download_and_extract() returns, not the package folder itself.
    """
    try:
        return griffe.load(package_name, search_paths=[str(search_path)])
    except Exception as e:
        # Common causes: the package is a C extension, the layout is unusual,
        # or the source uses syntax newer than your Python. Log and move on.
        print(f"griffe failed for {package_name}: {type(e).__name__}: {e}")
        return None


def diff_versions(package_name: str, old_path, new_path) -> list[dict]:
    """
    Return one row per candidate breaking change between two versions.

    Each row already carries the cheap structural features from Part 4 —
    these are free, and several of them turn out to be strong signals.
    """
    old = load_api(package_name, old_path)
    new = load_api(package_name, new_path)
    if old is None or new is None:
        return []

    rows = []
    for b in griffe.find_breaking_changes(old, new):
        symbol = b.obj.path              # e.g. "click.decorators.HelpOption"
        parts = symbol.split(".")

        rows.append(
            {
                "package": package_name,
                "symbol": symbol,
                # .name gives "OBJECT_REMOVED", not "BreakageKind.OBJECT_REMOVED"
                "kind": b.kind.name,
                "explanation": ANSI.sub("", b.explain()),
                # --- structural features (Part 4) ---
                # A leading underscore anywhere means the author considered it
                # private. Private things are fair game to change, so this is
                # a strong negative signal.
                #
                # KNOWN WEAKNESS, worth fixing in week 4 and worth mentioning
                # in an interview: this also flags dunders like __version__ and
                # __all__, which are public by convention. Run the demo on
                # jinja2 and you will see jinja2.__version__ marked private.
                # A better rule ignores names that start AND end with "__".
                # Leave it as-is for now — measure the fix, do not assume it.
                "is_private": any(p.startswith("_") for p in parts),
                # click.echo (depth 1) is used far more than
                # click.parser._OptionParser.add (depth 3).
                "module_depth": symbol.count("."),
                "is_top_level": symbol.count(".") == 1,
                # Longer names tend to be more obscure.
                "name_length": len(parts[-1]),
            }
        )

    return rows


if __name__ == "__main__":
    import pathlib
    import sys

    from download import download_and_extract, list_releases

    package = sys.argv[1] if len(sys.argv) > 1 else "click"

    releases = list_releases(package, last_n=6)
    old_rel, new_rel = releases[-2], releases[-1]

    base = pathlib.Path(f"data/sdists/{package}")
    old_path = download_and_extract(old_rel["url"], base / old_rel["version"])
    new_path = download_and_extract(new_rel["url"], base / new_rel["version"])

    rows = diff_versions(package, old_path, new_path)

    print(f"\n{package} {old_rel['version']} -> {new_rel['version']}")
    print(f"{len(rows)} candidate breaking changes\n")
    for r in rows[:15]:
        flag = "private" if r["is_private"] else "PUBLIC "
        print(f"  [{flag}] {r['kind']:<26} {r['symbol']}")
