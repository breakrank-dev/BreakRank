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
message ends "-> keyword-only"); R2's, a default changed from a plain
literal (PARAMETER_CHANGED_DEFAULT whose old value is a plain literal
and not a spelling-only change, fixer/rewrite.py's `plain` and
`cosmetic`) or a parameter made required (PARAMETER_CHANGED_REQUIRED).
A function counts once, however many rows or upgrades carry it; a
function the fixer would withhold on (a removal beside the moved
parameter) still counts, since the evaluation reads withheld lines too.
A function is identified by its defining path, so one that moved
modules between two upgrades counts once per module; the header says
how many paths reach more than one defining path.

A scanning package USES a function when one of its pairs names the
function's defining path or one of its export paths (the label's
matching, §24), or, for a constructor (`...__init__`), the class's path
by either spelling: the scan records `from redis import Redis` as
`redis.Redis`, and the fixer's locate takes a `Redis(...)` call to the
constructor's rows. A pair names a path, not a call: a class imported
for a subclass or an annotation counts as its constructor, so the count
is an upper bound on the call sites the fixer meets. What the scan
cannot record, a method called on an instance (`r.set(...)`), is not
counted. So the sample leans to constructors, module-level functions
and explicit class attributes; it is a convenience sample biased to
large consumers, and the report says so. Scanner names are PEP 503
names (`fixer.env.normalise`), spellings merged and the first kept for
display; a package is not downstream of itself: a pair whose scanner is
the changed package is dropped, as the scan drops self-use.

The ranking is by the number of distinct rewritable functions used,
ties by name; the first TEN are the sample; the next five are reserves,
taken in order only when a sample package's latest release cannot be
downloaded as source or the old version of the changed package cannot
be installed from a wheel, each substitution recorded in NOTES. One
entry per source repository: a second distribution built from the same
repository (mlflow-skinny beside mlflow, fastmcp-slim beside fastmcp;
PyPI names the same repository for both and ships them at the same
version) would put the same call sites in the sample twice, so it is
marked and skipped and the positions close up. `SAME_SOURCE` lists the
pairs; the data must agree, or the run stops: a second distribution can
only use what the first does. When the first is not ranked at all (its
scan failed, or it uses nothing rewritable) the second stands for the
repository, and a note says so, under the table and in the csv's note
column. A tie at either cut, the sample's or
the reserves', is said under the table, since there the name decides.
The printed table (the top 25, or through the last reserve, with each
package's count of functions, changed packages and upgrades), the ten's
detail (changed packages, upgrades, functions), and the pairs file's
row count, distinct count and sha256 go into NOTES before any run.
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

from fixer.bundle import version_key  # noqa: E402
from fixer.env import normalise  # noqa: E402
from fixer.rewrite import cosmetic, kind_change, plain  # noqa: E402

PAIRS = pathlib.Path("data/usage_pairs.csv")
BUNDLE = pathlib.Path("fixer/data/breakrank.sqlite")
OUT = pathlib.Path("data/eval_sample.csv")

R1_KINDS = {"PARAMETER_MOVED", "PARAMETER_CHANGED_KIND"}
R2_KINDS = {"PARAMETER_CHANGED_DEFAULT", "PARAMETER_CHANGED_REQUIRED"}

# Second distributions of one repository, as PyPI's project_urls name it
# (read 26 Oct; both of each pair at the same version that day). The
# sample takes one entry per repository (NOTES §38.16). The scan may
# have read the two at different versions; the subset check below is
# where that would show.
SAME_SOURCE = {
    "mlflow-skinny": "mlflow",      # github.com/mlflow/mlflow, 3.17.2
    "fastmcp-slim": "fastmcp",      # github.com/PrefectHQ/fastmcp, 4.1.3
}

SKIPPED = "same source as {}, skipped"


class SampleError(Exception):
    """The same-source table and the data disagree; nothing is written."""


class PairsError(Exception):
    """The pairs file is not the scan's shape; nothing is written."""


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
    kinds = sorted(R1_KINDS | R2_KINDS)
    rows = con.execute(
        "SELECT package, version_from, version_to, symbol, export_paths, "
        "kind, message, old_value, new_value FROM change WHERE kind IN "
        f"({', '.join('?' * len(kinds))})", kinds)
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


def shared_paths(by_path) -> list[str]:
    """The paths that reach more than one defining path: a function that
    moved modules between upgrades, counted once per module."""
    return sorted(p for p, keys in by_path.items() if len(keys) > 1)


def read_pairs(pairs: pathlib.Path):
    """(scanner, symbol) rows, both stripped, and the file's sha256. A
    short or empty row is refused with its line number."""
    digest = hashlib.sha256()
    out = []
    with pairs.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    with pairs.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "scanner" not in reader.fieldnames \
                or "symbol" not in reader.fieldnames:
            raise PairsError(f"{pairs}: expected the columns scanner and "
                             f"symbol, found {reader.fieldnames}")
        for row in reader:
            scanner = (row.get("scanner") or "").strip()
            symbol = (row.get("symbol") or "").strip()
            if not scanner or not symbol:
                raise PairsError(f"{pairs}:{reader.line_num}: a short or "
                                 "empty row; the scan writes two columns")
            out.append((scanner, symbol))
    return out, digest.hexdigest()


def distinct(pairs) -> int:
    """How many distinct (PEP 503 scanner, symbol) pairs the rows hold;
    the scan's file is appended to across resumed runs."""
    return len({(normalise(s), y) for s, y in pairs})


def rank(pairs, by_path):
    """scanner -> set of (package, function) it uses, self-use dropped,
    spellings merged by PEP 503 name (the first kept for display);
    ranked by the count, ties by name."""
    uses: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    spelling: dict[str, str] = {}
    for scanner, symbol in pairs:
        name = normalise(scanner)
        spelling.setdefault(name, scanner)
        for key in by_path.get(symbol, ()):
            if normalise(key[0]) == name:
                continue
            uses[name].add(key)
    ranked = sorted(uses.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [(spelling[name], keys) for name, keys in ranked]


def select(ranked, top: int, reserves: int = 5, same_source=None):
    """(scanner -> its role, scanner -> its note). A role is 'sample',
    'reserve', the skipped mark for a second distribution of a
    repository, or ''. The positions close up over the skipped. A second
    distribution whose first is not ranked stands, with a note. A table
    the data contradicts, or that lists a name on both sides, raises
    SampleError."""
    if same_source is None:
        same_source = SAME_SOURCE
    table = {normalise(a): normalise(p) for a, p in same_source.items()}
    written = {normalise(p): p for p in same_source.values()}
    both = sorted(set(table) & set(table.values()))
    if both:
        raise SampleError(
            f"{', '.join(both)}: listed both as a second distribution and "
            "as a first in the same-source table; one entry per repository "
            "names the first once")
    uses = {normalise(s): keys for s, keys in ranked}
    spelling = {normalise(s): s for s, _ in ranked}
    roles: dict[str, str] = {}
    notes: dict[str, str] = {}
    taken = 0
    for scanner, keys in ranked:
        name = normalise(scanner)
        standing = None
        if name in table:
            primary = table[name]
            if primary not in uses:
                standing = written[primary]
            else:
                extra = sorted(f for _, f in keys - uses[primary])
                if extra:
                    raise SampleError(
                        f"{scanner} is listed as a second distribution of "
                        f"{spelling[primary]}, but uses {len(extra)} "
                        f"function(s) {spelling[primary]} does not "
                        f"({', '.join(extra[:4])}): the scan read the two at "
                        "different versions, or the table is wrong; settle "
                        "which before running again")
                roles[scanner] = SKIPPED.format(spelling[primary])
                continue
        taken += 1
        roles[scanner] = ("sample" if taken <= top else
                          "reserve" if taken <= top + reserves else "")
        if standing is not None:
            where = ("stands for the repository; check which"
                     if roles[scanner] else
                     "is outside the sample and the reserves")
            notes[scanner] = (
                f"{scanner} is listed as a second distribution of {standing}, "
                "which is not ranked (its scan failed, or it uses nothing "
                f"rewritable): {scanner} {where}")
    return roles, notes


def describe(ranked, upgrades, rules, roles, top: int, show: int,
             notes=()) -> list[str]:
    lines = []
    lines.append(f"{'rank':>4}  {'functions':>9}  {'packages':>8}  "
                 f"{'upgrades':>8}  scanning package")
    places = [i for i, (s, _) in enumerate(ranked, 1)
              if roles[s] in ("sample", "reserve")]
    show = max(show, places[-1] if places else 0)
    for i, (scanner, keys) in enumerate(ranked[:show], 1):
        pkgs = {k[0] for k in keys}
        ups = {(k[0], u) for k in keys for u in upgrades[k]}
        mark = f"  <- {roles[scanner]}" if roles[scanner] else ""
        lines.append(f"{i:>4}  {len(keys):>9}  {len(pkgs):>8}  {len(ups):>8}"
                     f"  {scanner}{mark}")
    sampled = [i for i, (s, _) in enumerate(ranked) if roles[s] == "sample"]
    reserved = [i for i, (s, _) in enumerate(ranked) if roles[s] == "reserve"]
    for last, what, after in ((sampled, "sampled", "the first reserve"),
                              (reserved, "reserve", "the first left out")):
        if not last:
            continue
        i = last[-1]
        nxt = next((j for j in range(i + 1, len(ranked))
                    if not roles[ranked[j][0]].startswith("same source")),
                   None)
        if nxt is not None and len(ranked[nxt][1]) == len(ranked[i][1]):
            lines.append(f"the last {what}, {ranked[i][0]}, ties on "
                         f"{len(ranked[i][1])} with {ranked[nxt][0]}, "
                         f"{after}: by name")
    for note in dict(notes).values():
        lines.append(f"note: {note}")
    lines.append("")
    lines.append(f"the sample, the {len(sampled)} with the most rewritable "
                 "functions (ties by name; one entry per repository):")
    for n, i in enumerate(sampled, 1):
        scanner, keys = ranked[i]
        lines.append("")
        lines.append(f"  {n:>2}. {scanner}, rank {i + 1}: {len(keys)} "
                     "function(s)")
        by_pkg: dict[str, list[tuple[str, str]]] = (
            collections.defaultdict(list))
        for k in keys:
            by_pkg[k[0]].append(k)
        for pkg in sorted(by_pkg):
            ks = sorted(by_pkg[pkg], key=lambda k: k[1])
            ups = sorted({u for k in ks for u in upgrades[k]},
                         key=lambda u: (version_key(u[1]), version_key(u[0])))
            ups_text = ", ".join(f"{a} -> {b}" for a, b in ups[:6])
            if len(ups) > 6:
                ups_text += f" and {len(ups) - 6} more"
            lines.append(f"      {pkg}: {len(ks)} function(s) in {len(ups)} "
                         f"upgrade(s): {ups_text}")
            for k in ks[:8]:
                tags = "+".join(sorted(rules[k]))
                lines.append(f"          {k[1]}   [{tags}]")
            if len(ks) > 8:
                lines.append(f"          ... and {len(ks) - 8} more")
    return lines


def write_csv(path: pathlib.Path, ranked, upgrades, roles, notes) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "scanner", "functions", "packages", "upgrades",
                    "role", "note", "detail"])
        for i, (scanner, keys) in enumerate(ranked, 1):
            pkgs = sorted({k[0] for k in keys})
            ups = {(k[0], u) for k in keys for u in upgrades[k]}
            detail = ";".join(f"{k[0]}:{k[1]}" for k in sorted(keys))
            w.writerow([i, scanner, len(keys), len(pkgs), len(ups),
                        roles[scanner], notes.get(scanner, ""), detail])


def main(argv=None, same_source=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=pathlib.Path, default=PAIRS)
    ap.add_argument("--bundle", type=pathlib.Path, default=BUNDLE)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--show", type=int, default=25,
                    help="rows of the table, at least through the last "
                         "reserve (default 25)")
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
    try:
        pairs, digest = read_pairs(args.pairs)
        ranked = rank(pairs, by_path)
        roles, notes = select(ranked, args.top, same_source=same_source)
    except (PairsError, SampleError) as e:
        print(e, file=sys.stderr)
        return 2
    functions = {k for keys in by_path.values() for k in keys}
    print(f"{args.pairs}: {len(pairs):,} row(s), {distinct(pairs):,} "
          f"distinct pair(s), sha256 {digest}")
    print(f"{args.bundle}: {len(functions):,} rewritable function(s) in "
          f"{len({k[0] for k in functions})} package(s); "
          f"{len(shared_paths(by_path))} path(s) reach more than one "
          "defining path")
    print(f"{len(ranked):,} scanning package(s) use at least one; "
          f"{sum(len(k) for _, k in ranked):,} (package, function) uses in "
          "all\n")
    print("\n".join(describe(ranked, upgrades, rules, roles, args.top,
                             args.show, notes)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, ranked, upgrades, roles, notes)
    print(f"\n{args.out}: {len(ranked):,} row(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
