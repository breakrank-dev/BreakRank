"""
Remove breakage rows that are wrong or superseded. Nothing else.

    python scripts/db_prune.py            # show what it would do
    python scripts/db_prune.py --yes      # actually delete

WHY THIS IS A SEPARATE SCRIPT.

ml/db.py has no DELETE anywhere, deliberately — a loader that can remove
rows is a loader that can remove the wrong ones when a run half-fails. And
scripts/db_extras.py is documented READ ONLY, which is exactly what makes
it safe to point at a live database while you are still thinking. Neither
property should be given up to save a file.

So deletion lives here, on its own, and does nothing without --yes.

WHY DELETION IS NEEDED AT ALL. The loader upserts and never removes, so
the breakage table is the union of every run ever made. After the 12 Sep
re-ingest it held 23,128 rows against 22,905 sent. Those 223 extras are
three different things and only two of them should go:

  87  UN-IMPORTABLE ROOT (NOTES §9.5). `python.grpcio.grpc.StatusCode.OK`,
      `bindings.python.py_src.tokenizers.models.BPE.from_file`. The module
      detector walked into a build directory and treated it as a package.
      Nobody can import these paths, so they are wrong regardless of which
      run produced them — and they are being served as real findings.
      DELETING THEM IS A PATCH, NOT A FIX. The fix is at ingest: normalise
      the stem before testing it against NOT_THE_LIBRARY, and descend
      further into sdists that nest their package. Until that lands these
      come back on every load.

  56  SUPERSEDED. We re-analysed that exact release and did not produce
      this row. tokenizers is the clear case: the same 12 changes were
      once recorded under two wrong roots, and a clean re-extraction
      produced one. Within a release we just re-analysed, our row set is
      authoritative.

  80  AGED OUT — KEPT, and this is the interesting one. The release left
      the six-release window, so we no longer look that far back. But the
      change still happened: someone upgrading litellm 1.95 -> 1.97 wants
      to know what 1.96 broke, and our window is a property of OUR ingest,
      not of their upgrade. Deleting these would make the product worse to
      satisfy a tidiness rule. Accumulation is correct here.

Varad's pandas 2.2.0 fixtures fall into the third group and are therefore
never touched by this script. That is luck rather than design — the
classifier cannot tell a fixture from a real aged-out release — but the
outcome is right: his rows are his to remove.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import breakage_rows, connect, load_frames, release_rows  # noqa: E402

BAD_ROOTS = ("python.", "bindings.", "py_src.", "crates.", "_cffi_src.")

# A prune that wants to remove a large share of the table is not a prune,
# it is a symptom — a stale changes.csv, the wrong database, a half-failed
# ingest. Refuse and make a human look.
MAX_SHARE = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete un-importable and superseded breakage rows.")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete. Without it this only reports.")
    args = ap.parse_args()

    from sqlalchemy import text

    changes, _usage, _packages = load_frames()
    rels = release_rows(changes)
    fake = {(r["package"], r["version"]): i for i, r in enumerate(rels, 1)}
    _rows, keys, _dropped = breakage_rows(changes, fake)
    ours = set(keys)
    live_rel = (set(zip(changes["package"], changes["version_to"]))
                | set(zip(changes["package"], changes["version_from"])))

    engine = connect()
    with engine.connect() as conn:
        has_sub = bool(conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'breakage' AND column_name = 'sub_target'
        """)).fetchone())
        sub_sel = "b.sub_target" if has_sub else "''"
        found = conn.execute(text(f"""
            SELECT b.id, p.name, r.version, b.symbol_path, b.kind, {sub_sel}
            FROM breakage b
            JOIN release r ON r.id = b.release_id
            JOIN package p ON p.id = r.package_id
        """)).fetchall()

    total = len(found)
    bad, superseded, aged = [], [], []
    for bid, name, ver, sym, kind, sub in found:
        if (name, ver, sym, kind, sub or "") in ours:
            continue
        if str(sym).startswith(BAD_ROOTS):
            bad.append(bid)
        elif (name, ver) in live_rel:
            superseded.append(bid)
        else:
            aged.append(bid)

    doomed = bad + superseded
    print(f"\nbreakage rows in table: {total:,}")
    print(f"  un-importable root   {len(bad):>5}   delete")
    print(f"  superseded           {len(superseded):>5}   delete")
    print(f"  aged out of window   {len(aged):>5}   KEEP — still true")
    print(f"\nwould delete {len(doomed):,} rows "
          f"({len(doomed) / max(total, 1):.1%} of the table)")

    if not doomed:
        print("\nNothing to do.")
        return

    if len(doomed) / max(total, 1) > MAX_SHARE:
        sys.exit(f"\nREFUSING: that is more than {MAX_SHARE:.0%} of the table.\n"
                 "Something is wrong upstream — a stale changes.csv, the "
                 "wrong database,\nor a half-finished ingest. Check before "
                 "deleting anything.")

    if not args.yes:
        print("\nDry run. Nothing was deleted. Re-run with --yes to apply.")
        print("Read NOTES §9.5 first if the un-importable count is large — "
              "deleting\nthem here is a patch, and they return on the next "
              "load until the\ningest is fixed.")
        return

    # predictions carry a foreign key to breakage, so they go first.
    with engine.begin() as conn:
        n_pred = conn.execute(
            text("DELETE FROM prediction WHERE breakage_id = ANY(:ids)"),
            {"ids": doomed}).rowcount
        n_brk = conn.execute(
            text("DELETE FROM breakage WHERE id = ANY(:ids)"),
            {"ids": doomed}).rowcount

    print(f"\ndeleted {n_pred:,} prediction rows and {n_brk:,} breakage rows")
    print("Run scripts/db_extras.py to confirm only the aged-out rows and "
          "any\nfixtures remain.")


if __name__ == "__main__":
    main()
