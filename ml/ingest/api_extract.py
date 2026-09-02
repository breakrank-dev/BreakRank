"""
Step 3 — read a package's API with griffe, and diff two versions.

This is the step that produces your CANDIDATE CHANGES: every difference
griffe considers potentially breaking between version A and version B.
There will be roughly 200 per major release and about 5 will actually
matter — ranking them is the entire project.

Nothing here is machine learning yet. This is the machine that makes the data.
"""

import logging
import pathlib
import re

import griffe

# griffe prints a lot of "could not resolve alias" warnings on real packages.
# They are harmless — it just means one module referenced something it could
# not follow. Silence them so your own output stays readable.
logging.getLogger("griffe").setLevel(logging.ERROR)

# b.explain() returns text with terminal colour codes baked in. Those look
# like garbage in a CSV, so strip them.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# Folders that live next to the real code in an sdist but are not the library.
NOT_THE_LIBRARY = {
    "test", "tests", "testing", "doc", "docs", "example", "examples",
    "benchmark", "benchmarks", "script", "scripts", "tools", "build", "dist",
    "setup", "conftest", "noxfile", "tasks",
}


def _is_noise(stem: str) -> bool:
    """Test suites, docs and build scaffolding that sit beside the real code."""
    low = stem.lower()
    return (
        low in NOT_THE_LIBRARY
        or low.startswith(("test_", "tests_", "_test"))
        or low.endswith(("_test", "_tests"))
        or stem.startswith(".")
    )


def _normalise(name: str) -> str:
    """`typing-extensions` and `typing_extensions` are the same thing."""
    return re.sub(r"[-_.]", "", name).lower()


def find_import_names(search_path, package_name: str = "") -> list[str]:
    """
    What can you actually `import` from this folder?

    You cannot derive this from the PyPI package name, and the failure is
    silent if you try. The name on PyPI is the DISTRIBUTION name; the name
    you import is the MODULE name, and they disagree constantly:

        typing-extensions  ->  typing_extensions      (hyphen)
        python-dateutil    ->  dateutil               (nothing like it)
        pyyaml             ->  yaml
        beautifulsoup4     ->  bs4
        pillow             ->  PIL
        scikit-learn       ->  sklearn
        attrs              ->  attr AND attrs         (two of them)

    30% of the top 300 PyPI packages have a hyphen alone, and hyphens are
    not even legal in Python identifiers — so guessing the import name from
    the distribution name loses roughly a third of the dataset before you
    start.

    So don't guess. Look. Read the extracted source tree and report what is
    importable: directories holding an __init__.py, plus top-level .py files.

    Returns candidates BEST FIRST. "Best" means: the one whose name matches
    the distribution name once you ignore hyphens and underscores; failing
    that, the one with the most Python files, on the theory that the library
    is bigger than whatever else is sitting next to it (a vendored
    dependency, an `exercises/` folder).
    """
    root = pathlib.Path(search_path)
    if not root.is_dir():
        return []

    real, namespace = [], []
    for p in sorted(root.iterdir()):
        stem = p.stem
        if _is_noise(stem):
            continue

        if p.is_dir():
            if (p / "__init__.py").is_file():
                real.append((p.name, len(list(p.rglob("*.py")))))
            elif any(p.rglob("*.py")):
                # PEP 420 namespace package — no __init__.py, still importable.
                # `protobuf` ships google/protobuf/ exactly like this.
                namespace.append((p.name, len(list(p.rglob("*.py")))))
        elif p.suffix == ".py" and not stem.startswith("_"):
            real.append((stem, 1))      # single-file libraries, e.g. six.py

    candidates = real or namespace
    target = _normalise(package_name)
    candidates.sort(key=lambda c: (_normalise(c[0]) != target, -c[1]))
    return [name for name, _ in candidates]


def load_api(package_name: str, search_path) -> griffe.Module:
    """
    Read one version of a single module into a griffe object tree.

    Raises on failure. That is deliberate: an earlier version of this
    swallowed the error and returned None, which made diff_versions return
    an empty list — so a package that CRASHED looked identical to a package
    with no changes. `attrs` silently reported "0 candidate changes" for
    three version pairs when in fact griffe never loaded it at all.

    A failure you cannot see is worse than a failure. Let it raise; the
    caller decides whether to log and continue.
    """
    return griffe.load(package_name, search_paths=[str(search_path)])


def load_all_modules(search_path, modules: list[str]) -> griffe.ModulesCollection:
    """
    Load every top-level module of one distribution into ONE collection.

    This matters more than it looks. `attrs` ships two importable modules,
    `attrs` and `attr`, and attrs/__init__.py does `from attr import field`.
    Load `attrs` on its own and that alias points into a module griffe has
    never seen, so the first attempt to follow it raises AliasResolutionError
    and the whole package yields nothing. Loading both into a shared
    collection lets the alias resolve, and attrs goes from 0 rows to real data.

    Note we deliberately do NOT call loader.resolve_aliases() here. That
    walks and resolves everything eagerly, so one unresolvable alias
    anywhere kills the package. Leaving resolution lazy means aliases are
    followed only when something actually asks for them, which is all
    find_breaking_changes needs.
    """
    loader = griffe.GriffeLoader(search_paths=[str(search_path)])
    for module in modules:
        try:
            loader.load(module)
        except Exception:
            # One module of several failed to load. Keep the others —
            # a partly-loaded distribution still produces honest rows for
            # the modules that did load.
            continue
    return loader.modules_collection


def diff_versions(package_name: str, old_path, new_path,
                  module: str | None = None) -> list[dict]:
    """
    Return one row per candidate breaking change between two versions.

    `package_name` is the PyPI name and is what lands in the `package`
    column. `module` is what you actually import — pass it explicitly, or
    leave it None and it is detected from the new version's source tree.
    Those two are different for about a third of PyPI; see find_import_names.

    Each row already carries the cheap structural features from Part 4 —
    these are free, and several of them turn out to be strong signals.

    Raises if either version fails to load. Callers running in bulk should
    catch, record the reason, and move on — see ml/ingest/run_ingest.py.
    """
    modules = [module] if module else find_import_names(new_path, package_name)
    if not modules:
        raise RuntimeError(
            f"no importable Python module found in {new_path} — "
            "probably a compiled-only or non-Python distribution"
        )

    old_col = load_all_modules(old_path, modules)
    new_col = load_all_modules(new_path, modules)

    shared = [m for m in modules if m in old_col and m in new_col]
    if not shared:
        raise RuntimeError(
            f"griffe could not load any of {modules} in both versions"
        )

    breakages = []
    errors = []
    for m in shared:
        try:
            # list() BEFORE extend(), and this is not a style choice.
            # find_breaking_changes is a generator that walks members in
            # alphabetical order. numpy's cyclic aliases crash it partway,
            # and `breakages.extend(generator)` keeps everything yielded
            # before the crash — which for numpy 2.4.6 -> 2.5.0 meant 33
            # rows running acos, acosh, all ... block, and then nothing.
            # Not a sample of numpy: the first 3% of the alphabet.
            # Building the list separately means a crash discards the whole
            # pair. Missing data is honest; a biased prefix that looks like
            # coverage is not.
            found = list(griffe.find_breaking_changes(old_col[m], new_col[m]))
        except Exception as e:
            errors.append(f"{m}: {type(e).__name__}")
            continue
        breakages.extend(found)

    if errors and not breakages:
        raise RuntimeError("; ".join(errors))

    rows = []
    for b in breakages:
        symbol = b.obj.path              # e.g. "click.decorators.HelpOption"
        parts = symbol.split(".")

        # Big packages ship their test suite inside the installed package —
        # numpy.random.tests.test_extending.required_version turned up in a
        # 50-package run. Test code is not public API, nobody downstream
        # imports it, so every such row would be a guaranteed negative
        # padding out the dataset. Drop them at the source.
        if any(p == "tests" or p.startswith("test_") for p in parts):
            continue

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
    import sys

    from download import download_and_extract, list_releases

    package = sys.argv[1] if len(sys.argv) > 1 else "click"

    releases = list_releases(package, last_n=6)
    old_rel, new_rel = releases[-2], releases[-1]

    base = pathlib.Path(f"data/sdists/{package}")
    old_path = download_and_extract(old_rel["url"], base / old_rel["version"])
    new_path = download_and_extract(new_rel["url"], base / new_rel["version"])

    modules = find_import_names(new_path, package)
    print(f"\nPyPI name '{package}' -> importable as {modules or 'NOTHING FOUND'}")

    rows = diff_versions(package, old_path, new_path)

    print(f"{old_rel['version']} -> {new_rel['version']}: "
          f"{len(rows)} candidate breaking changes\n")
    for r in rows[:15]:
        flag = "private" if r["is_private"] else "PUBLIC "
        print(f"  [{flag}] {r['kind']:<26} {r['symbol']}")
