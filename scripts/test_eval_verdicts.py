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

Since 23 Nov the record names the reader, carries the suggestion's
reason and says which pins add a warning (NOTES §38.18). Expected,
written BEFORE that code ran. The reader is the split's: sheet numbers
1, 100, 201 and 215 are Vaibhav's, 101, 200, 216 and 230 Varad's, 0
and 231 nobody's (refused by number). A reason holding the mark
"[ADDS A WARNING" marks a pin that adds one; the same words without
the mark do not. A second sheet of six read calls:
  100 rewritten, no-op, reason with the mark      (Vaibhav)
  101 rewritten, no-op, reason with the mark      (Varad)
  102 rewritten, unsafe, reason with the mark     (Varad)
  103 rewritten, correct, reason without it       (Varad)
  215 withheld, nothing needed                    (Vaibhav)
  216 withheld, justified                         (Varad)
By reader: Vaibhav rewritten 1 read (no-op 1), withheld 1 read (nothing
needed 1); Varad rewritten 3 read (correct 1, no-op 1, unsafe 1),
withheld 1 read (justified 1). Warnings: 3 of the 4 rewritten calls
read add one, no-op 2 and unsafe 1. The record's rows carry reader,
reason and adds_warning ("yes" or empty). A sheet with call 231 is
refused with its number. 5 checks more, 11 in all.
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

        print("\nTHE READER, THE REASON AND THE WARNINGS (23 Nov)")

        def refused_by_number(i):
            try:
                ev.reader_of(i)
            except ev.SheetError as e:
                return f"#{i}" in str(e)
            return False

        check("the reader of a call is the split's: 1, 100, 201 and 215 the "
              "first reader's, 101, 200, 216 and 230 the second's; 0 and 231 "
              "have none and are refused by number",
              ([ev.reader_of(i)
                for i in (1, 100, 201, 215, 101, 200, 216, 230)],
               refused_by_number(0), refused_by_number(231),
               ev.reader_of("101")),
              (["Vaibhav"] * 4 + ["Varad"] * 4, True, True, "Varad"))
        mark = ("3.0 warns when copy is passed "
                "[ADDS A WARNING: pandas 3.0 warns]")
        check("a pin that adds a warning is told by the mark in its reason "
              "and by nothing else",
              (ev.adds_warning({"reason": mark}),
               ev.adds_warning({"reason": "it adds a warning, in words"}),
               ev.adds_warning({"reason": ""}), ev.adds_warning({})),
              (True, False, False, False))
        two = [row(100, "rewritten", "pa", "x", R2, "no-op"),
               row(101, "rewritten", "pa", "x", R2, "no-op"),
               row(102, "rewritten", "pb", "y", R2, "unsafe"),
               row(103, "rewritten", "pb", "y", R2, "correct"),
               row(215, "withheld", "pa", "x", R1W, "nothing needed"),
               row(216, "withheld", "pb", "y", R1W, "justified")]
        for r in two[:3]:
            r["reason"] = mark
        sheet2 = tmp / "sheet2.csv"
        write(two, sheet2)
        rows2 = ev.read_sheet(sheet2)
        by = ev.summarise_readers(rows2)
        check("the counts by reader, and the warnings beside the verdicts: "
              "3 of the 4 rewritten calls read add one",
              by,
              {"by_reader": {
                  "Vaibhav": {
                      "rewritten": {"read": 1, "correct": 0, "no-op": 1,
                                    "unsafe": 0, "wrong": 0},
                      "withheld": {"read": 1, "justified": 0,
                                   "not justified": 0,
                                   "nothing needed": 1}},
                  "Varad": {
                      "rewritten": {"read": 3, "correct": 1, "no-op": 1,
                                    "unsafe": 1, "wrong": 0},
                      "withheld": {"read": 1, "justified": 1,
                                   "not justified": 0,
                                   "nothing needed": 0}}},
               "warnings": {"read": 4, "adding": 3, "correct": 0,
                            "no-op": 2, "unsafe": 1, "wrong": 0}})
        out2 = io.StringIO()
        ev.print_readers(by, out2)
        text2 = out2.getvalue()
        check("the summary names each reader with his counts and gives the "
              "warnings as counts over counts",
              ("Vaibhav: rewritten 1 read: correct 0, no-op 1, unsafe 0, "
               "wrong 0" in text2,
               "Varad: rewritten 3 read: correct 1, no-op 1, unsafe 1, "
               "wrong 0" in text2,
               "withheld 1 read: justified 1, not justified 0, nothing "
               "needed 0" in text2,
               "add a warning under the new version: 3 of the 4 rewritten "
               "calls read" in text2,
               "%" in text2),
              (True, True, True, True, False))
        record2 = tmp / "verdicts2.csv"
        ev.write_record(rows2, record2)
        with record2.open(encoding="utf-8", newline="") as f:
            back2 = list(csv.DictReader(f))
        outside = tmp / "outside.csv"
        write(two + [row(231, "rewritten", "pa", "x", R2, "no-op")], outside)
        try:
            ev.read_sheet(outside)
            no_reader = None
        except ev.SheetError as e:
            no_reader = str(e)
        check("the record names the reader, carries the reason and says "
              "which calls add a warning; a call outside the split is "
              "refused with its number",
              ([(r["id"], r["reader"], r["adds_warning"], r["reason"] == mark)
                for r in back2],
               no_reader is not None and "#231" in no_reader),
              ([("100", "Vaibhav", "yes", True), ("101", "Varad", "yes", True),
                ("102", "Varad", "yes", True), ("103", "Varad", "", False),
                ("215", "Vaibhav", "", False), ("216", "Varad", "", False)],
               True))

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. The verdicts are read, counted and recorded")
    print("as §38.18 says.")


if __name__ == "__main__":
    main()
