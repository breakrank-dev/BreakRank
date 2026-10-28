"""
Does scripts/eval_verdicts.py turn the filled sheet into §38.18's
record and its counts over counts, and refuse a sheet that is not fully
or not validly filled?

    python scripts/test_eval_verdicts.py

Expected, written BEFORE the code ran. A sheet of five read calls:
  1 rewritten, R1 and R2 both acted, verdict correct   (sample pa, pkg x)
  2 rewritten, R1 withheld and R2 pinned, verdict no-op (pa, x)
  3 rewritten, R1 alone, verdict wrong, with a note     (pb, y)
  4 withheld, verdict justified                          (pa, x)
  5 withheld, verdict nothing needed                     (pb, y)
Counts: rewritten read 3 of a total of 10 (7 unread): correct 1, no-op
1, unsafe 0, wrong 1; by rule, R1 on calls 1 and 3: correct 1, wrong 1;
R2 on calls 1 and 2: correct 1, no-op 1; by sample package, pa: 2 read,
correct 1, no-op 1; pb: 1 read, wrong 1. Withheld read 2 of 4:
justified 1, not justified 0, nothing needed 1. A verdict outside the
four (or the three) is refused with the call's id; an empty verdict is
refused as unread; a verdict is read whatever its case and spacing
("No-Op", "nothing  needed"). The record written has one row per read
call and no context column. 6 checks.
"""

import csv
import io
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scripts.eval_verdicts as ev  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name, got, want):
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}\n          want {want!r}")
        failures.append(name)


R1 = "R1, pass by name: port 6380 -> port=6380"
R1W = "R1 withheld: b was positional-only in the old version"
R2 = "R2, pin the old default: retries=3 (now 5); delete an argument"


def row(i, kind, sample, pkg, rules, verdict, note=""):
    return {"id": str(i), "kind": kind, "sample": sample, "changed": pkg,
            "from": "1", "to": "2", "file": "f.py", "line": str(i),
            "column": "1", "confidence": "resolved", "symbol": "x.f",
            "before": "f()", "after": "f(a=1)", "rules": rules,
            "old_signature": "f(a=1)", "new_signature": "f(a=2)",
            "context": "ctx", "suggested": "correct", "reason": "r",
            "verdict": verdict, "note": note}


SHEET = [row(1, "rewritten", "pa", "x", f"{R1} || {R2}", "correct"),
         row(2, "rewritten", "pa", "x", f"{R1W} || {R2}", "No-Op"),
         row(3, "rewritten", "pb", "y", R1, "wrong", "pins a sentinel"),
         row(4, "withheld", "pa", "x", R1W, "justified"),
         row(5, "withheld", "pb", "y", R1W, "nothing  needed")]


def write(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        sheet = tmp / "sheet.csv"
        write(SHEET, sheet)
        rows = ev.read_sheet(sheet)
        check("the rules a rewritten call carries: the lines that say the "
              "rule acted, not the withheld ones",
              [ev.rules_applied(r) for r in rows],
              [["R1", "R2"], ["R2"], ["R1"], [], []])
        counts = ev.summarise(rows, total_rewritten=10, total_withheld=4)
        check("the counts over counts: overall, by rule, by sample package, "
              "the withheld three, the unread",
              counts,
              {"rewritten": {"read": 3, "total": 10, "unread": 7,
                             "correct": 1, "no-op": 1, "unsafe": 0,
                             "wrong": 1},
               "by_rule": {"R1": {"read": 2, "correct": 1, "no-op": 0,
                                  "unsafe": 0, "wrong": 1},
                           "R2": {"read": 2, "correct": 1, "no-op": 1,
                                  "unsafe": 0, "wrong": 0}},
               "by_sample": {"pa": {"read": 2, "correct": 1, "no-op": 1,
                                    "unsafe": 0, "wrong": 0},
                             "pb": {"read": 1, "correct": 0, "no-op": 0,
                                    "unsafe": 0, "wrong": 1}},
               "withheld": {"read": 2, "total": 4, "unread": 2,
                            "justified": 1, "not justified": 0,
                            "nothing needed": 1}})
        out = io.StringIO()
        ev.print_summary(counts, out)
        text = out.getvalue()
        check("the summary says counts over counts, never a rate alone",
              ("rewritten calls read: 3 of 10" in text,
               "correct 1, no-op 1, unsafe 0, wrong 1" in text,
               "R1: 2 read" in text and "R2: 2 read" in text,
               "withheld calls read: 2 of 4" in text,
               "%" in text),
              (True, True, True, True, False))
        record = tmp / "verdicts.csv"
        ev.write_record(rows, record)
        with record.open(encoding="utf-8", newline="") as f:
            back = list(csv.DictReader(f))
        check("the record: one row per read call, the verdict as written in "
              "the four or three words, no context column",
              ([(r["id"], r["verdict"], r["note"]) for r in back],
               "context" in back[0], "rules" in back[0]),
              ([("1", "correct", ""), ("2", "no-op", ""),
                ("3", "wrong", "pins a sentinel"), ("4", "justified", ""),
                ("5", "nothing needed", "")], False, True))
        bad = tmp / "bad.csv"
        write(SHEET[:2] + [row(3, "rewritten", "pb", "y", R1, "ok")], bad)
        try:
            ev.read_sheet(bad)
            refused = None
        except ev.SheetError as e:
            refused = str(e)
        empty = tmp / "empty.csv"
        write(SHEET[:1] + [row(2, "withheld", "pa", "x", R1W, "")], empty)
        try:
            ev.read_sheet(empty)
            unread = None
        except ev.SheetError as e:
            unread = str(e)
        check("a verdict outside the words is refused with the call's id; an "
              "empty one is refused as unread",
              (refused is not None and "#3" in refused and "ok" in refused,
               unread is not None and "#2" in unread and "unread" in unread),
              (True, True))
        check("a withheld call cannot take a rewritten call's verdict",
              ev.valid("withheld", "correct"), False)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The verdicts are read, counted and recorded")
    print("as §38.18 says.")


if __name__ == "__main__":
    main()
