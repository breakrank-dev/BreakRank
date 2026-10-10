"""
Does the inherited-member fold do what it claims, and only that?

    python scripts/test_inherited.py

No network, no sdists, no changes.csv. It writes tiny packages into a temp
directory, diffs them with the real diff_collections(), and checks the row
counts. Runs in under a second, so it can be run before every ingest.

WHY THIS FILE EXISTS.

griffe's diff walks `old_obj.all_members`, and for a class that is
`{**inherited_members, **members}`. So a method removed from a base class
is compared once per SUBCLASS, and each comparison yields its own breakage
with the subclass's path. transformers 5.16.1 -> 5.17.0 dropped three
methods from ModuleUtilsMixin; 3,242 model classes inherit it; griffe
reported 9,729 removals. That was 30% of the whole 32,405-row dataset,
sitting on three lines of someone else's refactor.

None of those rows is false. They are just not 9,729 events.

fold_inherited() collapses them onto the class that defines the method and
records `inherited_by`. The three cases below are the three ways that can
go wrong, and each is a test:

  1. The fold happens at all, and the count is right.
  2. THE BASE IS PRIVATE. griffe skips private members, so no row for the
     defining method exists — folding onto it would DELETE a real public
     breakage. The fold must keep a public subclass instead.
  3. IT DOES NOT OVER-FOLD. An ordinary removal, and a class that declares
     its own method, must come through untouched. A fold that quietly ate
     normal rows would shrink the dataset and look like success.

Plus one property that is not about correctness but about the database:
the stand-in chosen in case 2 must be the SAME one on every run, because
breakage is keyed on symbol_path and an unstable choice writes a second
row instead of upserting the first.
"""

import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import griffe  # noqa: E402

from ml.ingest.api_extract import diff_collections  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def build(root: pathlib.Path, base_name: str, has_method: bool,
          n_subclasses: int = 10) -> pathlib.Path:
    """A package with one mixin and N classes that inherit it."""
    pkg = root / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "models").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")

    body = "    def gone(self):\n        return 1\n" if has_method else "    pass\n"
    module = "_base" if base_name.startswith("_") else "base"
    (pkg / f"{module}.py").write_text(f"class {base_name}:\n{body}")

    for i in range(n_subclasses):
        (pkg / "models" / f"m{i}.py").write_text(
            f"from ..{module} import {base_name}\n\n\n"
            f"class M{i}({base_name}):\n    pass\n")
    return root


def rows_for(old_root: pathlib.Path, new_root: pathlib.Path) -> list[dict]:
    old = {"pkg": griffe.load("pkg", search_paths=[str(old_root)],
                              allow_inspection=False)}
    new = {"pkg": griffe.load("pkg", search_paths=[str(new_root)],
                              allow_inspection=False)}
    return diff_collections("pkg", old, new, ["pkg"])


def case_public_base(tmp: pathlib.Path) -> None:
    print("\n1. PUBLIC BASE — one removal, ten subclasses")
    a, b = tmp / "pub_a", tmp / "pub_b"
    build(a, "Mixin", has_method=True)
    build(b, "Mixin", has_method=False)

    raw = list(griffe.find_breaking_changes(
        griffe.load("pkg", search_paths=[str(a)], allow_inspection=False),
        griffe.load("pkg", search_paths=[str(b)], allow_inspection=False)))
    print(f"     griffe itself reports {len(raw)} breakages for 1 change")
    check("griffe amplifies (this is the bug being fixed)", len(raw), 11)

    rows = rows_for(a, b)
    check("folds to one row", len(rows), 1)
    check("row sits on the DEFINING class", rows[0]["symbol"],
          "pkg.base.Mixin.gone")
    check("inherited_by counts the subclasses", rows[0]["inherited_by"], 10)


def case_private_base(tmp: pathlib.Path) -> None:
    print("\n2. PRIVATE BASE — the change must survive the fold")
    a, b = tmp / "priv_a", tmp / "priv_b"
    build(a, "_Mixin", has_method=True)
    build(b, "_Mixin", has_method=False)

    rows = rows_for(a, b)
    check("kept exactly one stand-in", len(rows), 1)
    check("stand-in is a PUBLIC subclass", rows[0]["symbol"],
          "pkg.models.m0.M0.gone")
    check("stand-in is not marked private", rows[0]["is_private"], False)
    check("count still recorded", rows[0]["inherited_by"], 10)

    picks = {rows_for(a, b)[0]["symbol"] for _ in range(5)}
    check("stand-in is the same on every run", len(picks), 1)


def case_no_over_fold(tmp: pathlib.Path) -> None:
    print("\n3. ORDINARY CHANGES — untouched")
    a, b = tmp / "plain_a", tmp / "plain_b"
    for root, keep in ((a, True), (b, False)):
        pkg = root / "pkg"
        if pkg.exists():
            shutil.rmtree(pkg)
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        # A module-level function, and a class that DECLARES its own method
        # while also inheriting one. Only the inherited copy may fold.
        (pkg / "base.py").write_text(
            "class Mixin:\n    def kept(self):\n        return 1\n")
        lines = ["from .base import Mixin", "", "", "def helper():",
                 "    return 1", "", "", "class Own(Mixin):"]
        lines += ["    def mine(self):", "        return 1"] if keep else ["    pass"]
        (pkg / "api.py").write_text("\n".join(lines) + "\n")
        if not keep:
            (pkg / "api.py").write_text(
                (pkg / "api.py").read_text().replace(
                    "def helper():\n    return 1", "def helper2():\n    return 1"))

    rows = rows_for(a, b)
    got = sorted((r["symbol"], r["inherited_by"]) for r in rows)
    print(f"     rows: {got}")
    check("plain function removal survives",
          ("pkg.api.helper", 0) in got, True)
    check("a class's OWN method removal survives",
          ("pkg.api.Own.mine", 0) in got, True)
    check("nothing else was invented or eaten", len(got), 2)


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-inherit-"))
    try:
        case_public_base(tmp)
        case_private_base(tmp)
        case_no_over_fold(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The fold collapses inherited repeats, keeps")
    print("the change when the base is private, and leaves normal rows alone.")


if __name__ == "__main__":
    main()
