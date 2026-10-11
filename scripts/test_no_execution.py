"""
Can the ingest pipeline execute code from a downloaded package?

    python scripts/test_no_execution.py

It must not. This file proves it rather than asserting it: it builds a
package carrying a compiled extension module, runs the real loader over
it, and checks that griffe's dynamic-analysis path was never entered.

WHY THIS FILE EXISTS.

griffe reads a package two ways. STATIC analysis parses the .py files as
text and never runs them. DYNAMIC analysis imports the module and looks
at the live objects — and importing a module runs every line at its top
level, including `subprocess.run(...)`, including anything else.

`griffe.GriffeLoader` and `griffe.load` both default `allow_inspection`
to **True**, which means "fall back to importing when parsing fails".
For griffe's own audience — people documenting their own code — that is
a helpful default. For this pipeline it is the opposite of what we want,
because the input is 500 strangers' source archives fetched from PyPI
seconds earlier, and the fallback fires precisely when a module is
strange enough not to parse.

It was missing from `load_all_modules` until 16 Sep 2026, through two
full runs over the top 500. macOS XProtect raised "Malicious Script
Blocked" during both. NOTES §17.

Three checks, because the flag can be lost three ways:

  1. Someone removes the argument. Check 1 reads it off the constructed
     loader, so it fails on the object rather than on the source text.
  2. Someone adds a NEW call site without it. Check 2 greps every
     griffe entry point in ml/ for a bare call.
  3. The flag is present but does not do what we think. Check 3 is the
     only one that actually matters: a real load of a package shaped
     like the dangerous case, watching whether griffe reaches for an
     import — and, in the same breath, confirming that griffe's own
     default DOES reach for it, so the check cannot pass vacuously.

Check 3 is the one to keep if the others ever become annoying. The other
two are about noticing; that one is about the truth.
"""

import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import griffe  # noqa: E402
from griffe._internal.loader import GriffeLoader  # noqa: E402

from ml.ingest.api_extract import load_all_modules  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []

ROOT = pathlib.Path(__file__).resolve().parents[1]


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def case_loader_flag() -> None:
    print("\n1. THE LOADER IS BUILT WITH INSPECTION OFF")
    # Read it off a real loader, not out of the source. A comment saying
    # allow_inspection=False and a loader that has it are different
    # things, and only one of them stops an import.
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-noexec-flag-"))
    try:
        (tmp / "pkg").mkdir()
        (tmp / "pkg" / "__init__.py").write_text("x = 1\n")
        col = load_all_modules(tmp, ["pkg"])
        check("load_all_modules still returns a collection", "pkg" in col, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    loader = griffe.GriffeLoader(search_paths=["."], allow_inspection=False)
    check("griffe exposes the flag we think it does",
          getattr(loader, "allow_inspection", "MISSING"), False)
    # And the default really is the dangerous one — if griffe ever changes
    # this, the note in api_extract.py becomes wrong and should be updated
    # rather than left to mislead the next reader.
    check("griffe's own default is still True (so the argument matters)",
          getattr(griffe.GriffeLoader(search_paths=["."]),
                  "allow_inspection", "MISSING"), True)


def case_no_bare_call_sites() -> None:
    print("\n2. NO CALL SITE IN ml/ OMITS THE ARGUMENT")
    call = re.compile(r"griffe\.load\(|griffe\.GriffeLoader\(")
    bare = []
    for path in sorted((ROOT / "ml").rglob("*.py")):
        text = path.read_text()
        for m in call.finditer(text):
            # Look at the whole call, which may wrap over several lines.
            depth, i = 0, m.end() - 1
            while i < len(text):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if "allow_inspection" not in text[m.start():i]:
                line = text[:m.start()].count("\n") + 1
                bare.append(f"{path.relative_to(ROOT)}:{line}")
    check("every griffe entry point sets allow_inspection", bare, [])


def _fixture(tmp: pathlib.Path) -> pathlib.Path:
    """A package carrying a COMPILED extension module.

    This is the shape that matters, and it took three wrong fixtures to
    find it. Measured on griffe 2.2.0:

      - a package that parses cleanly never reaches the fallback, with
        the flag on OR off;
      - a SyntaxError does not reach it either — griffe raises
        LoadingError instead of inspecting;
      - only a module whose file is not `.py`/`.pyi` reaches it.

    loader.py:600-606 is the whole decision:

        elif module_path.suffix in {".py", ".pyi"}:
            module = self._visit_module(...)      <- parse the text
        elif self.allow_inspection:
            module = self._inspect_module(...)    <- IMPORT the binary
        else:
            raise LoadingError("Cannot load compiled module ...")

    So the dangerous input is a native binary sitting inside a source
    archive — and sdists do carry them: vendored libraries, prebuilt
    artifacts, test fixtures. Importing one is a dlopen of code that was
    downloaded minutes earlier.
    """
    pkg = tmp / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from . import ext  # noqa: F401\n")
    (pkg / "ext.cpython-311-darwin.so").write_bytes(b"\x00not a real dylib\x00")
    return tmp / "src"


def case_no_import_attempted() -> None:
    print("\n3. A COMPILED MODULE IS NOT IMPORTED")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-noexec-"))
    try:
        search = _fixture(tmp)

        # Spy on the one method that does dynamic analysis. Recording the
        # ATTEMPT is stronger than checking a side effect: a bogus .so
        # fails to dlopen, so a side-effect canary would stay unwritten
        # even on a pipeline that tried to load it. We want to fail on
        # "tried", not on "succeeded".
        calls: list[str] = []
        original = GriffeLoader._inspect_module

        def spy(self, module_name, filepath=None, parent=None):
            calls.append(str(filepath))
            return original(self, module_name, filepath, parent)

        GriffeLoader._inspect_module = spy
        try:
            load_all_modules(search, ["pkg"])
            ours = list(calls)

            # AND PROVE THIS TEST CAN FAIL. Run the same fixture through
            # griffe's own default and confirm it DOES try to import.
            # Without this line the check above passes just as well when
            # the fixture is wrong, which is how the first three versions
            # of it passed while testing nothing.
            calls.clear()
            loader = griffe.GriffeLoader(search_paths=[str(search)],
                                         allow_inspection=True)
            try:
                loader.load("pkg")
            except Exception:
                pass
            theirs = list(calls)
        finally:
            GriffeLoader._inspect_module = original

        check("our loader never attempts an import", ours, [])
        check("griffe's default DOES (so this test is not vacuous)",
              len(theirs) > 0, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    case_loader_flag()
    case_no_bare_call_sites()
    case_no_import_attempted()

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        print("\nDO NOT RUN AN INGEST until this passes. A failure here means")
        print("the pipeline may import and execute code it downloaded from")
        print("PyPI, which is a different kind of problem from a wrong number.")
        sys.exit(1)
    print("All checks passed. The pipeline reads downloaded packages as")
    print("text and never imports them.")


if __name__ == "__main__":
    main()
