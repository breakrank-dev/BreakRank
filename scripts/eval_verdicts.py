"""
Week 5's verdicts (NOTES §38.18): the filled sheet read back, checked,
counted as counts over counts, and written as the committed record.

    python scripts/eval_verdicts.py
    python scripts/eval_verdicts.py --sheet data/eval/sheet.csv --counts data/eval/counts.csv --record docs/eval/verdicts.csv

Reads  data/eval/sheet.csv     scripts/eval_sheet.py's sheet with the
                               author's verdict (and note) filled in
       data/eval/counts.csv    scripts/eval_run.py count: the totals the
                               read calls are a part of
Writes docs/eval/verdicts.csv  one row per read call: where it is, the
                               rule lines as printed and which rules acted,
                               what the sheet suggested, the author's
                               verdict and note; no source text

A rewritten call's verdict is one of correct, no-op, unsafe, wrong; a
withheld call's one of justified, not justified, nothing needed (§38.6,
§38.18), read whatever the case and spacing. An empty verdict is a call
not read, and the sheet is refused until every call has one; a word
outside the lists is refused with the call's id. A rewritten call
counts in the tally of every rule whose line under it says the rule
acted ("R1, pass by name:" or "R2, pin the old default:"), so a call
both rules rewrote counts twice, said so in the summary.
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re
import sys

REWRITTEN = ("correct", "no-op", "unsafe", "wrong")
WITHHELD = ("justified", "not justified", "nothing needed")
WORDS = {"rewritten": REWRITTEN, "withheld": WITHHELD}
ACTED = {"R1": "R1, pass by name:", "R2": "R2, pin the old default:"}
RECORD = ["id", "kind", "sample", "changed", "from", "to", "file", "line",
          "column", "confidence", "symbol", "rules", "rules_applied",
          "suggested", "verdict", "note"]


class SheetError(Exception):
    """The sheet is not filled as the rule asks; nothing is recorded."""


def normal(word: str) -> str:
    return re.sub(r"[\s_-]+", " ", word.strip().lower()).replace(
        "no op", "no-op")


def valid(kind: str, verdict: str) -> bool:
    return normal(verdict) in WORDS.get(kind, ())


def rules_applied(row: dict) -> list[str]:
    """The rules whose line under the call says the rule acted."""
    lines = [ln.strip() for ln in row["rules"].split(" || ")]
    return [tag for tag, start in ACTED.items()
            if row["kind"] == "rewritten"
            and any(ln.startswith(start) for ln in lines)]


def read_sheet(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            word = r.get("verdict", "")
            if not word.strip():
                raise SheetError(
                    f"#{r['id']} ({r['file']}:{r['line']}) has no verdict: "
                    "unread; every drawn call is read before the counts "
                    "are made")
            if not valid(r["kind"], word):
                raise SheetError(
                    f"#{r['id']} ({r['file']}:{r['line']}) has the verdict "
                    f"{word!r}, not one of {', '.join(WORDS[r['kind']])}")
            r["verdict"] = normal(word)
            rows.append(r)
    return rows


def _tally(rows) -> dict:
    t = {"read": len(rows)}
    for w in REWRITTEN:
        t[w] = sum(1 for r in rows if r["verdict"] == w)
    return t


def summarise(rows: list[dict], total_rewritten: int,
              total_withheld: int) -> dict:
    rw = [r for r in rows if r["kind"] == "rewritten"]
    wh = [r for r in rows if r["kind"] == "withheld"]
    out = {"rewritten": {**{"read": len(rw), "total": total_rewritten,
                            "unread": total_rewritten - len(rw)},
                         **{w: sum(1 for r in rw if r["verdict"] == w)
                            for w in REWRITTEN}},
           "by_rule": {}, "by_sample": {},
           "withheld": {**{"read": len(wh), "total": total_withheld,
                           "unread": total_withheld - len(wh)},
                        **{w: sum(1 for r in wh if r["verdict"] == w)
                           for w in WITHHELD}}}
    for tag in ACTED:
        out["by_rule"][tag] = _tally([r for r in rw
                                      if tag in rules_applied(r)])
    for s in sorted({r["sample"] for r in rw}):
        out["by_sample"][s] = _tally([r for r in rw if r["sample"] == s])
    return out


def print_summary(c: dict, out=sys.stdout) -> None:
    rw, wh = c["rewritten"], c["withheld"]
    print(f"rewritten calls read: {rw['read']} of {rw['total']} "
          f"({rw['unread']} unread): correct {rw['correct']}, no-op "
          f"{rw['no-op']}, unsafe {rw['unsafe']}, wrong {rw['wrong']}",
          file=out)
    for tag, t in c["by_rule"].items():
        print(f"  {tag}: {t['read']} read: correct {t['correct']}, no-op "
              f"{t['no-op']}, unsafe {t['unsafe']}, wrong {t['wrong']}",
              file=out)
    print("  (a call both rules rewrote counts once above and in both rules' "
          "lines)", file=out)
    for s, t in c["by_sample"].items():
        print(f"  {s}: {t['read']} read: correct {t['correct']}, no-op "
              f"{t['no-op']}, unsafe {t['unsafe']}, wrong {t['wrong']}",
              file=out)
    print(f"withheld calls read: {wh['read']} of {wh['total']} "
          f"({wh['unread']} unread): justified {wh['justified']}, not "
          f"justified {wh['not justified']}, nothing needed "
          f"{wh['nothing needed']}", file=out)


def write_record(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECORD)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k, "") for k in RECORD},
                        "rules_applied": "+".join(rules_applied(r))})


def totals(counts_csv: pathlib.Path) -> tuple[int, int]:
    rw = wh = 0
    with counts_csv.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rw += int(r["rewritten"] or 0)
            wh += int(r["withheld"] or 0)
    return rw, wh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", type=pathlib.Path,
                    default=pathlib.Path("data/eval/sheet.csv"))
    ap.add_argument("--counts", type=pathlib.Path,
                    default=pathlib.Path("data/eval/counts.csv"))
    ap.add_argument("--record", type=pathlib.Path,
                    default=pathlib.Path("docs/eval/verdicts.csv"))
    args = ap.parse_args(argv)
    try:
        rows = read_sheet(args.sheet)
    except SheetError as e:
        print(e, file=sys.stderr)
        return 2
    rw, wh = totals(args.counts)
    c = summarise(rows, rw, wh)
    print_summary(c)
    write_record(rows, args.record)
    overruled = [r for r in rows
                 if r.get("suggested")
                 and normal(r["suggested"]) != r["verdict"]]
    print(f"{len(overruled)} verdict(s) differ from the sheet's suggestion"
          + (": " + ", ".join(f"#{r['id']}" for r in overruled)
             if overruled else ""))
    print(f"{args.record}: {len(rows)} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
