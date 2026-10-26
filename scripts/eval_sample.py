"""
Week 5's sample (NOTES §38.6, §38.16): the ten downstream packages the
fixer is evaluated on, chosen by a rule fixed before this ran, from the
usage scan and the bundle. Nothing is downloaded here; the run itself is
week 5's.

    python scripts/eval_sample.py
    python scripts/eval_sample.py --pairs data/usage_pairs.csv --top 10

Reads  data/usage_pairs.csv          one row per (scanning package, symbol it
                                     uses), ml/ingest/run_usage.py's output
       fixer/data/breakrank.sqlite   the bundle: every change with its paths
Writes data/eval_sample.csv          one row per scanning package that uses
                                     a rewritable function, ranked

The rule. A function is REWRITABLE when the bundle holds a row on it of
a kind the fixer's two rules act on: R1's, a parameter moved
(PARAMETER_MOVED) or made keyword-only (PARAMETER_CHANGED_KIND whose
message ends "-> keyword-only"); R2's, a default changed to or from a
plain literal (PARAMETER_CHANGED_DEFAULT whose old value is a plain
literal and not a spelling-only change, fixer/rewrite.py's `plain` and
`cosmetic`) or a parameter made required (PARAMETER_CHANGED_REQUIRED).
A function counts once, however many rows or upgrades carry it; a
function the fixer would withhold on (a removal beside the moved
parameter) still counts, since the evaluation reads withheld lines too.

A scanning package USES a function when one of its pairs names the
function's defining path or one of its export paths (the label's
matching, §24), or, for a constructor (`...__init__`), the class's path
by either spelling: the scan records `from redis import Redis` as
`redis.Redis`, and the fixer's locate takes a `Redis(...)` call to the
constructor's rows. What the scan cannot record, a method called on an
instance (`r.set(...)`), is not counted, so the sample leans to
constructors, module-level functions and explicit class attributes; it
is a convenience sample biased to large consumers, and the report says
so. A package is not downstream of itself: a pair whose scanner is the
changed package (PEP 503 names) is dropped, as the scan drops self-use.

The ranking is by the number of distinct rewritable functions used,
ties by name; the first TEN are the sample; the next five are reserves,
taken in order only when a sample package's latest release cannot be
downloaded as source or the old version of the changed package cannot
be installed from a wheel, each substitution recorded in NOTES. The
printed table (the top 25, with each package's count of functions,
changed packages and upgrades), the ten's detail (changed packages,
upgrades, functions), and the pairs file's row count and sha256 go into
NOTES before any run.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fixer.env import normalise  # noqa: E402
from fixer.rewrite import cosmetic, kind_change, plain  # noqa: E402

PAIRS = pathlib.Path("data/usage_pairs.csv")
BUNDLE = pathlib.Path("fixer/data/breakrank.sqlite")
OUT = pathlib.Path("data/eval_sample.csv")

R1_KINDS = {"PARAMETER_MOVED", "PARAMETER_CHANGED_KIND"}
R2_KINDS = {"PARAMETER_CHANGED_DEFAULT", "PARAMETER_CHANGED_REQUIRED"}


def rewritable(kind: str, message: str, old: str | None,
               new: str | None) -> str | None:
    """'R1' or 'R2' when the row is one the fixer's rules act on, else
    None."""
    if kind == "PARAMETER_MOVED":
        return "R1"
    if kind == "PARAMETER_CHANGED_KIND":
        kinds = kind_change(message)
        return "R1" if kinds and kinds[1] == "keyword-only" else None
    if kind == "PARAMETER_CHANGED_REQUIRED":
        return "R2"
    if kind == "PARAMETER_CHANGED_DEFAULT":
        if old is None or new is None or not plain(old) or cosmetic(old, new):
            return None
        return "R2"
    return None


def paths_of(symbol: str, exports: str) -> set[str]:
    """Every path a downstream package can write for the function: the
    defining path, its export paths, and for a constructor the class's
    paths too."""
    out = {symbol} | {p for p in exports.split(";") if p}
    if symbol.endswith(".__init__"):
        out |= {p[:-len(".__init__")] for p in out if p.endswith(".__init__")}
    return out


def load_functions(bundle: pathlib.Path):
    """path -> set of (package, function); and (package, function) ->
    {(version_from, version_to)}, from the bundle's rewritable rows."""
    con = sqlite3.connect(bundle)
    by_path: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    upgrades: dict[tuple[str, str], set[tuple[str, str]]] = (
        collections.defaultdict(set))
    rules: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    rows = con.execute(
        "SELECT package, version_from, version_to, symbol, export_paths, "
        "kind, message, old_value, new_value FROM change WHERE kind IN "
        "('PARAMETER_MOVED', 'PARAMETER_CHANGED_KIND', "
        "'PARAMETER_CHANGED_DEFAULT', 'PARAMETER_CHANGED_REQUIRED')")
    for pkg, frm, to, symbol, exports, kind, message, old, new in rows:
        rule = rewritable(kind, message, old, new)
        if rule is None:
            continue
        key = (pkg, symbol)
        upgrades[key].add((frm, to))
        rules[key].add(rule)
        for p in paths_of(symbol, exports or ""):
            by_path[p].add(key)
    con.close()
    return by_path, upgrades, rules


def read_pairs(pairs: pathlib.Path):
    """(scanner, symbol) pairs, the row count and the file's sha256."""
    digest = hashlib.sha256()
    out = []
    with pairs.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    with pairs.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "scanner" not in reader.fieldnames \
                or "symbol" not in reader.fieldnames:
            raise SystemExit(f"{pairs}: expected the columns scanner and "
                             f"symbol, found {reader.fieldnames}")
        for row in reader:
            out.append((row["scanner"], row["symbol"]))
    return out, digest.hexdigest()


def rank(pairs, by_path):
    """scanner -> set of (package, function) it uses, self-use dropped;
    ranked by the count, ties by name."""
    uses: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for scanner, symbol in pairs:
        for key in by_path.get(symbol, ()):
            if normalise(key[0]) == normalise(scanner):
                continue
            uses[scanner].add(key)
    ranked = sorted(uses.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return ranked


def describe(ranked, upgrades, rules, top: int, show: int,
             reserves: int = 5) -> list[str]:
    lines = []
    lines.append(f"{'rank':>4}  {'functions':>9}  {'packages':>8}  "
                 f"{'upgrades':>8}  scanning package")
    for i, (scanner, keys) in enumerate(ranked[:show], 1):
        pkgs = {k[0] for k in keys}
        ups = {(k[0], u) for k in keys for u in upgrades[k]}
        mark = ("  <- sample" if i <= top else
                "  <- reserve" if i <= top + reserves else "")
        lines.append(f"{i:>4}  {len(keys):>9}  {len(pkgs):>8}  {len(ups):>8}"
                     f"  {scanner}{mark}")
    lines.append("")
    lines.append(f"the sample, the {top} with the most rewritable functions "
                 "(ties by name):")
    for i, (scanner, keys) in enumerate(ranked[:top], 1):
        lines.append("")
        lines.append(f"  {i:>2}. {scanner}: {len(keys)} function(s)")
        by_pkg: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for k in keys:
            by_pkg[k[0]].append(k)
        for pkg in sorted(by_pkg):
            ks = sorted(by_pkg[pkg], key=lambda k: k[1])
            ups = sorted({u for k in ks for u in upgrades[k]},
                         key=lambda u: (u[1], u[0]))
            ups_text = ", ".join(f"{a} -> {b}" for a, b in ups[:6])
            if len(ups) > 6:
                ups_text += f" and {len(ups) - 6} more"
            lines.append(f"      {pkg}: {len(ks)} function(s) in {len(ups)} "
                         f"upgrade(s): {ups_text}")
            for k in ks[:8]:
                lines.append(f"          {k[1]}   [{'+'.join(sorted(rules[k]))}]")
            if len(ks) > 8:
                lines.append(f"          ... and {len(ks) - 8} more")
    return lines


def write_csv(path: pathlib.Path, ranked, upgrades) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "scanner", "functions", "packages", "upgrades",
                    "detail"])
        for i, (scanner, keys) in enumerate(ranked, 1):
            pkgs = sorted({k[0] for k in keys})
            ups = {(k[0], u) for k in keys for u in upgrades[k]}
            detail = ";".join(f"{k[0]}:{k[1]}" for k in sorted(keys))
            w.writerow([i, scanner, len(keys), len(pkgs), len(ups), detail])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=pathlib.Path, default=PAIRS)
    ap.add_argument("--bundle", type=pathlib.Path, default=BUNDLE)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args(argv)
    if not args.pairs.is_file():
        print(f"{args.pairs} is not a file; the usage scan writes it "
              "(ml/ingest/run_usage.py)", file=sys.stderr)
        return 2
    if not args.bundle.is_file():
        print(f"{args.bundle} is not a file; scripts/build_bundle.py builds "
              "it", file=sys.stderr)
        return 2
    by_path, upgrades, rules = load_functions(args.bundle)
    pairs, digest = read_pairs(args.pairs)
    ranked = rank(pairs, by_path)
    functions = {k for keys in by_path.values() for k in keys}
    print(f"{args.pairs}: {len(pairs):,} pairs, sha256 {digest[:12]}; "
          f"{args.bundle}: {len(functions):,} rewritable function(s) in "
          f"{len({k[0] for k in functions})} package(s)")
    print(f"{len(ranked):,} scanning package(s) use at least one; "
          f"{sum(len(k) for _, k in ranked):,} (package, function) uses in "
          "all\n")
    print("\n".join(describe(ranked, upgrades, rules, args.top, args.show)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, ranked, upgrades)
    print(f"\n{args.out}: {len(ranked):,} row(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
