"""
The reading sheets of week 5 (NOTES §38.18): for every call the draw
named, what the reader needs beside it, from the harness's files alone.

    python scripts/eval_sheet.py
    python scripts/eval_sheet.py --root data/eval

Reads  data/eval/draw.csv            the drawn calls (scripts/eval_run.py draw)
       data/eval/<sample>/<p>-<from>-<to>.txt   the reports
       data/eval/manifest.json       where each sample's tree is
       data/eval/old/<p>-<v>, new/   the two versions, for the signatures
Writes data/eval/sheets.txt          one numbered block per call, to read
       data/eval/sheet.csv           one row per call, with the columns
                                     suggested, reason, verdict, note empty

Each block: the call as the report found it and as the fixer would
write it; the rule lines the report printed under it; the symbol's
signature in the old version and in the new, read statically with
griffe from the two directories (fixer/oldapi.py, the same reading the
fixer did; "not read" when a version cannot answer); and the three
source lines around the call. The verdicts are §38.18's: for a
rewritten call CORRECT, NO-OP, UNSAFE or WRONG; for a withheld one
JUSTIFIED, NOT JUSTIFIED or NOTHING NEEDED. The sheet suggests one; the
author's verdict is the one recorded (scripts/eval_verdicts.py).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fixer.env import EnvError, find_installed  # noqa: E402
from fixer.oldapi import OldApi, Param  # noqa: E402
import scripts.eval_run as er  # noqa: E402

SEP = " || "   # between rule lines; the report never prints "||"
VERDICTS = {"rewritten": "correct / no-op / unsafe / wrong",
            "withheld": "justified / not justified / nothing needed"}
FIELDS = ["id", "kind", "sample", "changed", "from", "to", "file", "line",
          "column", "confidence", "symbol", "before", "after", "rules",
          "old_signature", "new_signature", "context", "suggested",
          "reason", "verdict", "note"]


def render(name: str, params: list[Param] | None) -> str:
    """The signature as Python writes it, or 'not read'."""
    if params is None:
        return "not read"
    parts = []
    seen_star = False
    for i, p in enumerate(params):
        text = p.name if p.default is None else f"{p.name}={p.default}"
        if p.kind == "var_positional":
            text, seen_star = f"*{p.name}", True
        elif p.kind == "var_keyword":
            text = f"**{p.name}"
        elif p.kind == "keyword_only" and not seen_star:
            parts.append("*")
            seen_star = True
        parts.append(text)
        if p.kind == "positional_only" and (
                i + 1 == len(params)
                or params[i + 1].kind != "positional_only"):
            parts.append("/")
    return f"{name}({', '.join(parts)})"


def api_for(pkg: str, env: pathlib.Path) -> OldApi | None:
    try:
        installed = find_installed(pkg, env)
    except EnvError:
        return None
    return OldApi(installed.import_roots, installed.dist_info.parent)


def signature(api: OldApi | None, symbol: str) -> str:
    name = symbol.rsplit(".", 1)[-1]
    if api is None:
        return "not read"
    return render(name, api.parameters(symbol))


def first_read(api: OldApi | None, symbols: list[str]) -> str:
    """The signature under the first of the call's symbols the version
    can answer (the fixer took the first the old version could show,
    fixer/rewrite.py's decide), else 'not read'."""
    for symbol in symbols:
        text = signature(api, symbol)
        if text != "not read":
            return text
    return "not read"


def context(tree: pathlib.Path, file: str, line: int) -> str:
    path = tree / file
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    lo, hi = max(1, line - 1), min(len(lines), line + 1)
    return "\n".join(f"{n:>6} | {lines[n - 1]}" for n in range(lo, hi + 1))


def read_draw(root: pathlib.Path) -> list[tuple]:
    """The drawn calls: (sample, changed, from, to, file, line, column,
    use, call, kind), a call told from another on its line by its use
    and its text (scripts/eval_run.py draw)."""
    with (root / "draw.csv").open(encoding="utf-8", newline="") as f:
        return [(r["sample"], r["changed"], r["from"], r["to"], r["file"],
                 int(r["line"]), int(r["column"]), r["use"], r["call"],
                 r["symbol"], r["kind"])
                for r in csv.DictReader(f)]


def _decisions_of(root: pathlib.Path, run) -> dict:
    sample, pkg, frm, to = run
    path = root / sample / f"{pkg}-{frm}-{to}.txt"
    if not path.is_file():
        return {}
    return {(d["file"], d["line"], d["column"], d["use"], d["call"],
             d["symbol"]): d
            for d in er.decisions(path.read_text(encoding="utf-8"))}


def missing(root: pathlib.Path, drawn) -> list[str]:
    """The drawn calls no report holds, as 'run file:line:column'."""
    out = []
    cache: dict = {}
    for sample, pkg, frm, to, file, line, col, use, call, symbol, *_ in drawn:
        run = (sample, pkg, frm, to)
        if run not in cache:
            cache[run] = _decisions_of(root, run)
        if (file, line, col, use, call, symbol) not in cache[run]:
            out.append(f"{er.key(run)} {file}:{line}:{col}")
    return out


def build(root: pathlib.Path, envs: dict) -> list[dict]:
    """One row per drawn call; envs maps (package, version) to the
    directory holding that version."""
    manifest = er.read_manifest(root)
    drawn = read_draw(root)
    gone = missing(root, drawn)
    if gone:
        raise SystemExit("drawn calls no report holds: " + "; ".join(gone))
    apis: dict = {}
    decisions: dict = {}
    rows = []
    for n, (sample, pkg, frm, to, file, line, col, use, call, symbol,
            kind) in enumerate(drawn, 1):
        run = (sample, pkg, frm, to)
        if run not in decisions:
            decisions[run] = _decisions_of(root, run)
        d = decisions[run][(file, line, col, use, call, symbol)]
        for version in (frm, to):
            if (pkg, version) not in apis:
                env = envs.get((pkg, version))
                apis[(pkg, version)] = api_for(pkg, env) if env else None
        tree = manifest["trees"].get(sample, {}).get("tree")
        rows.append({
            "id": n, "kind": kind, "sample": sample, "changed": pkg,
            "from": frm, "to": to, "file": file, "line": line, "column": col,
            "confidence": d["confidence"],
            "symbol": " / ".join(d["symbols"]),
            "before": d["code"], "after": d["call"],
            "rules": SEP.join(d["rules"]),
            "old_signature": first_read(apis[(pkg, frm)], d["symbols"]),
            "new_signature": first_read(apis[(pkg, to)], d["symbols"]),
            "context": context(pathlib.Path(tree), file, line) if tree else "",
            "suggested": "", "reason": "", "verdict": "", "note": ""})
    return rows


def block(r: dict) -> str:
    lines = [f"#{r['id']}  {r['kind']}   {r['sample']} / {r['changed']} "
             f"{r['from']} -> {r['to']}   {r['file']}:{r['line']}:"
             f"{r['column']}   {r['confidence']}",
             f"  symbol   {r['symbol']}",
             f"  before   {r['before']}"]
    if r["after"]:
        lines.append(f"  after    {r['after']}")
    for rule in r["rules"].split(SEP):
        if rule:
            lines.append(f"  {rule}")
    lines.append(f"  old ({r['changed']} {r['from']})  {r['old_signature']}")
    lines.append(f"  new ({r['changed']} {r['to']})  {r['new_signature']}")
    if r["context"]:
        lines.append("  source")
        lines.extend("   " + ln for ln in r["context"].splitlines())
    lines.append(f"  suggested: {r['suggested'] or '-'}"
                 + (f"   reason: {r['reason']}" if r["reason"] else ""))
    lines.append(f"  verdict ({VERDICTS[r['kind']]}): {r['verdict'] or ''}"
                 + (f"   note: {r['note']}" if r["note"] else ""))
    return "\n".join(lines)


def write(root: pathlib.Path, rows: list[dict]):
    text = root / "sheets.txt"
    out_csv = root / "sheet.csv"
    head = (f"{len(rows)} call(s) to read (NOTES §38.18): "
            f"{sum(1 for r in rows if r['kind'] == 'rewritten')} rewritten, "
            f"{sum(1 for r in rows if r['kind'] == 'withheld')} withheld. "
            "For each, the verdict goes into sheet.csv's verdict column, a "
            "reason into note when the suggestion is overruled.\n")
    text.write_text(head + "\n" + "\n\n".join(block(r) for r in rows) + "\n",
                    encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return text, out_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=er.ROOT)
    args = ap.parse_args(argv)
    envs = {}
    for stage in ("old", "new"):
        base = args.root / stage
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and "-" in d.name:
                    pkg, version = d.name.rsplit("-", 1)
                    envs[(pkg, version)] = d
    rows = build(args.root, envs)
    text, out_csv = write(args.root, rows)
    unread = sum(1 for r in rows
                 if "not read" in (r["old_signature"], r["new_signature"]))
    print(f"{len(rows)} call(s) on the sheet; {unread} with a signature not "
          f"read; {text}; {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
