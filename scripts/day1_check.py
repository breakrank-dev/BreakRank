"""
Day 1 environment check.

Run this AFTER you have installed everything:

    python scripts/day1_check.py

It tells you, in plain English, whether each piece of your setup works.
If every line says OK, you are ready to build. If something says FAIL,
the message tells you what to do about it.

This script makes two small internet requests. That is intentional —
half the things that break on day 1 are network problems, not code problems.
"""

import sys
import platform

PASS = "  OK   "
FAIL = " FAIL  "
WARN = " WARN  "

results = []


def check(name, fn, fix):
    """Run one check, print the result, remember whether it passed."""
    try:
        detail = fn()
        print(f"[{PASS}] {name}  ->  {detail}")
        results.append(True)
    except Exception as e:
        print(f"[{FAIL}] {name}")
        print(f"          reason: {type(e).__name__}: {e}")
        print(f"          fix:    {fix}")
        results.append(False)


# ---------------------------------------------------------------- 1. Python
def _python():
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        raise RuntimeError(
            f"you are on Python {major}.{minor}; this project needs 3.11 or newer"
        )
    return f"Python {platform.python_version()} on {platform.system()}"


check(
    "Python version is 3.11+",
    _python,
    "On macOS: brew install python@3.12, then delete .venv and create it again "
    "with: python3.12 -m venv .venv",
)


# ------------------------------------------------- 2. Running inside the venv
def _venv():
    # sys.prefix differs from sys.base_prefix only inside a virtual environment.
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("not running inside a virtual environment")
    return sys.prefix


check(
    "Running inside the .venv",
    _venv,
    "Run:  source .venv/bin/activate    (your prompt should then start with '(.venv)')",
)


# ------------------------------------------------------------- 3. The imports
def _imports():
    import griffe, httpx, tenacity, packaging, pandas, numpy, sklearn, matplotlib  # noqa
    return "griffe, httpx, tenacity, packaging, pandas, numpy, scikit-learn, matplotlib"


check(
    "Core libraries import",
    _imports,
    "Run:  pip install -r requirements.txt",
)


# ------------------------------------------- 4. LightGBM (the macOS trap)
def _lightgbm():
    import lightgbm
    return f"lightgbm {lightgbm.__version__}"


check(
    "LightGBM imports",
    _lightgbm,
    "On macOS LightGBM needs Apple's OpenMP runtime. Run:  brew install libomp  "
    "then:  pip install --force-reinstall lightgbm",
)


# --------------------------------------------------- 5. griffe actually works
def _griffe_works():
    import griffe
    # 'json' is part of Python itself, so this needs no download.
    pkg = griffe.load("json")
    n = len(list(pkg.members))
    if n == 0:
        raise RuntimeError("griffe loaded the json module but found no members")
    return f"read {n} members out of Python's built-in json module"


check(
    "griffe can read a package",
    _griffe_works,
    "Run:  pip install --upgrade griffe",
)


# ------------------------------------------------------- 6. PyPI is reachable
def _pypi():
    import httpx
    r = httpx.get("https://pypi.org/pypi/requests/json", timeout=30)
    r.raise_for_status()
    return f"pypi.org answered, 'requests' has {len(r.json()['releases'])} releases"


check(
    "PyPI API is reachable",
    _pypi,
    "Check your internet. If you are on college wifi, try a phone hotspot — "
    "some campus networks block package downloads.",
)


# --------------------------------------------------------------- 7. Git + gh
def _git():
    import subprocess
    v = subprocess.run(["git", "--version"], capture_output=True, text=True)
    if v.returncode != 0:
        raise RuntimeError("git not found")
    name = subprocess.run(
        ["git", "config", "--get", "user.name"], capture_output=True, text=True
    ).stdout.strip()
    if not name:
        raise RuntimeError("git works but your name is not configured")
    return f"{v.stdout.strip()}, committing as '{name}'"


check(
    "git is installed and configured",
    _git,
    'Run:  git config --global user.name "Your Name"  and  '
    'git config --global user.email "you@example.com"',
)


# ------------------------------------------------------------------- Summary
print()
if all(results):
    print("=" * 62)
    print("  Everything passed. You are ready to build.")
    print("  Next:  python ml/ingest/packages.py")
    print("=" * 62)
else:
    failed = results.count(False)
    print("=" * 62)
    print(f"  {failed} check(s) failed. Fix those first — read the 'fix:' lines above.")
    print("  Nothing later in the project will work until these pass.")
    print("=" * 62)
    sys.exit(1)
