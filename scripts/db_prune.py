"""
Remove breakage rows that are wrong or superseded. Nothing else.

    python scripts/db_prune.py            # show what it would do
    python scripts/db_prune.py --yes      # actually delete

WHY THIS IS A SEPARATE SCRIPT.

ml/db.py deletes no breakage row, deliberately — a loader that can remove
rows is a loader that can remove the wrong ones when a run half-fails.
(Its one DELETE, added 1 Oct, replaces a model version's scores in the
same transaction that writes the new ones, and refuses a run with under
half as many; NOTES §28.) And
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
      authoritative. "Re-analysed" is read from both files (NOTES §33.9,
      rule A, from 18 Oct): a version_to of changes.csv, or a release an
      agreeing releases.csv gives analysed_clean, since the ingest writes
      that only after the pair was diffed and nothing came of it.

  80  AGED OUT — KEPT, and this is the interesting one. The release left
      the six-release window, so we no longer look that far back. But the
      change still happened: someone upgrading litellm 1.95 -> 1.97 wants
      to know what 1.96 broke, and our window is a property of OUR ingest,
      not of their upgrade. Deleting these would make the product worse to
      satisfy a tidiness rule. Accumulation is correct here.

Varad's fixtures fall into the third group and are therefore never
touched by this script: since migration 005 they are the package
breakrank-fixture, which no changes.csv contains. That is luck rather
than design — the classifier cannot tell a fixture from a real aged-out
release — but the outcome is right: his rows are his to remove.

AFTER A DELETE, THE RELEASES RECOUNT (NOTES §33.7, 15 Oct). release.n_changes
is a denormalised count the loader writes from its files, and until today
a prune left it as it was, so a release that lost rows here kept a count
of rows that were gone (§33.6 found 159 releases whose count was not
their rows, from every cause, this among them). Now, in the same
transaction as the delete, every release that lost a row has n_changes
set from the table, where the database has the column (migration 005),
and a release left with no row at all is printed by name and NOT
relabelled: an `analysed` one that today's files do not name becomes
`unknown` on the next run of scripts/db_repair_counts.py (§33.7, rule
2), and one today's files name is the loader's.

A FOURTH GROUP, COUNTED AND NEVER DELETED HERE (NOTES §33.5, 12 Oct): the
version-string and package-metadata rows, F1's and F39's, which the
loader no longer writes and scripts/db_drop_versions.py removes with a
dry run and an --expect count of its own. They are 6.5% of the table by
the loader's count, over this script's guard, which is why they have a
script of their own. Until 12 Oct this script could not see them: the
ones in today's changes.csv counted as "ours" and were skipped without
a word, and the ones from earlier loads were split between superseded
and aged out. Now they are named first, as what they are, whatever else
they would be. Decided by ml/db.py's not_a_change(), build.py's two
rules, from the explanation the table keeps as detail.griffe_message.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import (FIXTURE_PACKAGE, breakage_rows, connect,  # noqa: E402
                   live_releases, load_frames, load_releases, not_a_change,
                   release_rows)

BAD_ROOTS = ("python.", "bindings.", "py_src.", "crates.", "_cffi_src.")

# A prune that wants to remove a large share of the table is not a prune,
# it is a symptom — a stale changes.csv, the wrong database, a half-failed
# ingest. Refuse and make a human look.
MAX_SHARE = 0.05


def classify(found: list, ours: set, live_rel: set) -> dict:
    """Every table row into one of four lists of ids. `found` rows are
    (id, package, version, symbol_path, kind, sub_target, griffe_message).

    The version-string and package-metadata rows come first, whatever
    else they are: they go through scripts/db_drop_versions.py, not here,
    and as there Varad's fixture rows are not among them. Then, for a row
    the current changes.csv does not have: an un-importable root is wrong
    wherever it came from; a row of a release the current run re-analysed
    is superseded; the rest aged out of the window and are kept (NOTES
    §8), the fixture's rows with them. A row may carry more fields after
    the seventh (main() adds release_id); only the first seven are read
    here."""
    out = {"versions": [], "bad": [], "superseded": [], "aged": []}
    if not found:
        return out
    frame = pd.DataFrame([tuple(r)[:7] for r in found],
                         columns=["id", "package", "version", "symbol",
                                  "kind", "sub_target", "explanation"])
    frame["explanation"] = frame["explanation"].fillna("").astype(str)
    marked = not_a_change(frame)
    for is_marked, row in zip(marked, found):
        bid, name, ver, sym, kind, sub, _msg = tuple(row)[:7]
        if is_marked and name != FIXTURE_PACKAGE:
            out["versions"].append(bid)
        elif (name, ver, sym, kind, sub or "") in ours:
            continue
        elif str(sym).startswith(BAD_ROOTS):
            out["bad"].append(bid)
        elif (name, ver) in live_rel:
            out["superseded"].append(bid)
        else:
            out["aged"].append(bid)
    return out


def recount(conn, rids: list) -> tuple[int, list]:
    """n_changes from the table for the releases that lost rows, and the
    ones left with none, (name, version, status), for a decision."""
    from sqlalchemy import text

    if not rids:
        return 0, []
    n = conn.execute(text("""
        UPDATE release SET n_changes = (SELECT count(*) FROM breakage b
                                        WHERE b.release_id = release.id)
        WHERE id = ANY(:rids)"""), {"rids": rids}).rowcount
    emptied = conn.execute(text("""
        SELECT p.name, r.version, r.analysis_status FROM release r
        JOIN package p ON p.id = r.package_id
        WHERE r.id = ANY(:rids) AND r.n_changes = 0
        ORDER BY p.name, r.version"""), {"rids": rids}).fetchall()
    return n, [(str(a), str(b), str(c)) for a, b, c in emptied]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete un-importable and superseded breakage rows.")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete. Without it this only reports.")
    # %% because argparse %-formats help text: a bare "5%)" made --help
    # crash with "unsupported format character" until 2 Oct.
    ap.add_argument("--max-share", type=float, default=MAX_SHARE,
                    help=f"refuse if the prune exceeds this share of the "
                         f"table (default {MAX_SHARE * 100:.0f}%%). Raise "
                         "it ONLY after reading the classification and "
                         "agreeing with every category.")
    args = ap.parse_args()

    from sqlalchemy import text

    changes, _usage, _packages = load_frames()
    rels = release_rows(changes)
    fake = {(r["package"], r["version"]): i for i, r in enumerate(rels, 1)}
    _rows, keys, _dropped = breakage_rows(changes, fake)
    ours = set(keys)
    # A BREAKAGE ROW HANGS OFF version_to, NOT off the release generally:
    # breakage_rows() attaches every row via release_id[(package,
    # version_to)], so a row for release V was produced by the pair
    # (previous -> V).
    #
    # So "superseded" can only mean: we re-ran THAT PAIR and it did not
    # produce this row. That requires V to be a version_to in the current
    # changes.csv. If V appears only as a version_from — which is what
    # happens when the six-release window rolls forward and V becomes the
    # OLDEST version in it — then the pair (previous -> V) was never
    # re-analysed, and we know nothing about whether its rows are still
    # correct.
    #
    # Until 16 Sep this set was version_to UNION version_from, which made
    # every such row look superseded. Measured on the 16 Sep load: that
    # classed 4,198 rows for deletion, including things like
    # `anthropic.AI_PROMPT was removed in 1.0.0` — a true, public,
    # module-level finding that nothing in this run contradicts. Deleting
    # it would have silently made the product worse to satisfy a
    # tidiness rule.
    #
    # Aged-out rows are KEPT (NOTES §8): someone upgrading litellm
    # 1.95 -> 1.97 still wants to know what 1.96 broke, and our window is
    # a property of OUR ingest, not of their upgrade.
    #
    # And from 18 Oct (NOTES §33.9, rule A) the pair counts as re-run
    # when releases.csv says so too: a release it gives analysed_clean,
    # in a package the two files agree on, was diffed against the one
    # before it and nothing came of it, so changes.csv has no row to hold
    # it by, and until today its earlier rows passed as aged out
    # (importlib-metadata 9.0.0, typedload 2.36; §33.8). no_baseline,
    # analysis_failed, no_source and yanked say the pair was NOT run, and
    # stay out: no_baseline is the version_from-only case above by name.
    # Read from the loader's own plan, in ml/db.py, so both read the files
    # the same way.
    live_rel = live_releases(changes, load_releases())

    engine = connect()
    with engine.connect() as conn:
        has_sub = bool(conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'breakage' AND column_name = 'sub_target'
        """)).fetchone())
        # release.n_changes exists from migration 005; without it there is
        # nothing to recount, and naming the column would abort the delete.
        has_count = bool(conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'release' AND column_name = 'n_changes'
        """)).fetchone())
        sub_sel = "b.sub_target" if has_sub else "''"
        found = conn.execute(text(f"""
            SELECT b.id, p.name, r.version, b.symbol_path, b.kind, {sub_sel},
                   b.detail ->> 'griffe_message', b.release_id
            FROM breakage b
            JOIN release r ON r.id = b.release_id
            JOIN package p ON p.id = r.package_id
        """)).fetchall()

    total = len(found)
    groups = classify(found, ours, live_rel)
    bad, superseded, aged = (groups["bad"], groups["superseded"],
                             groups["aged"])

    doomed = bad + superseded
    print(f"\nbreakage rows in table: {total:,}")
    print(f"  un-importable root   {len(bad):>5}   delete")
    print(f"  superseded           {len(superseded):>5}   delete")
    print(f"  aged out of window   {len(aged):>5}   KEEP — still true")
    print(f"  version strings and  {len(groups['versions']):>5}   NOT THIS "
          "SCRIPT'S: scripts/db_drop_versions.py removes\n  package metadata"
          f"{'':12}these (F1, F39; NOTES §33.5), with its own dry run")
    print(f"\nwould delete {len(doomed):,} rows "
          f"({len(doomed) / max(total, 1):.1%} of the table)")

    if not doomed:
        print("\nNothing to do.")
        return

    share = len(doomed) / max(total, 1)
    if share > args.max_share:
        # THE GUARD IS THE POINT, so raising it has to be a decision
        # somebody makes in a command, not a constant somebody edits.
        #
        # There is one legitimate way to exceed 5%: a change to the
        # EXTRACTION that moves symbol paths. The inherited-member fold
        # (NOTES §10.2) did exactly that — a change previously recorded
        # once per subclass is now recorded once on the defining class, so
        # every old path is superseded at once. Those rows are wrong and
        # should go. That is still a human decision, because "a large
        # share of the table is superseded" and "changes.csv is stale or
        # this is the wrong database" produce an identical number here.
        sys.exit(
            f"\nREFUSING: {share:.1%} of the table, over the "
            f"{args.max_share:.0%} limit.\n\n"
            "Two things produce this number and only one is safe:\n"
            "  1. A stale changes.csv, the wrong database, or a "
            "half-finished ingest.\n"
            "  2. An extraction change that moved symbol paths, so many "
            "rows are\n     genuinely superseded at once.\n\n"
            "Run scripts/db_extras.py and read the classification. If every "
            "doomed\nrow is 'superseded' or 'un-importable' and you know "
            "which extraction\nchange caused it, re-run with "
            f"--max-share {min(share + 0.05, 1.0):.2f} --yes.")

    if not args.yes:
        print("\nDry run. Nothing was deleted. Re-run with --yes to apply.")
        print("Read NOTES §9.5 first if the un-importable count is large — "
              "deleting\nthem here is a patch, and they return on the next "
              "load until the\ningest is fixed.")
        return

    # predictions carry a foreign key to breakage, so they go first; then
    # the releases that lost rows are recounted (§33.7), same transaction.
    gone = set(doomed)
    rids = sorted({int(tuple(r)[7]) for r in found if tuple(r)[0] in gone})
    with engine.begin() as conn:
        n_pred = conn.execute(
            text("DELETE FROM prediction WHERE breakage_id = ANY(:ids)"),
            {"ids": doomed}).rowcount
        n_brk = conn.execute(
            text("DELETE FROM breakage WHERE id = ANY(:ids)"),
            {"ids": doomed}).rowcount
        n_rel, emptied = recount(conn, rids) if has_count else (0, [])

    print(f"\ndeleted {n_pred:,} prediction rows and {n_brk:,} breakage rows;")
    if has_count:
        print(f"n_changes set from the table on {n_rel:,} release(s) that "
              "lost rows.")
    else:
        print("no n_changes column (migration 005 is not applied), so "
              "nothing was recounted.")
    if emptied:
        print(f"** {len(emptied):,} release(s) are left with no row at all. "
              "Their status is NOT changed here:\n** an `analysed` one that "
              "today's files do not name becomes `unknown` on the next\n** "
              "run of scripts/db_repair_counts.py (NOTES §33.7, rule 2); one "
              "today's files name\n** is the loader's.")
        for name, ver, status in emptied:
            print(f"**   {name} {ver}   {status}")
    print("Run scripts/db_extras.py to confirm only the aged-out rows and "
          "any\nfixtures remain.")


if __name__ == "__main__":
    main()
