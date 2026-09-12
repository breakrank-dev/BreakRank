"""
What is in the breakage table that we did not put there?

    python scripts/db_extras.py

READ ONLY. Every statement in this file is a SELECT. It will not insert,
update or delete anything, so it is safe to run against the live database
while you are still deciding what to do.

WHY THIS EXISTS. The loader reports "23,024 sent, 23,030 in table". Those
two numbers measure different things: the first is how many rows we built
from changes.csv, the second is a count of the whole table. Six rows we
never sent were already sitting there. That is not corruption — every row
we sent landed — but six rows of unknown origin will be served by the API
as if they were real findings, and a demo that shows a made-up breaking
change is worse than one that shows none.

The likely explanation is boring: Varad needed rows to build the API
against before the pipeline produced any, so he wrote a few by hand. This
script confirms or refutes that instead of assuming it.

It also checks the other direction — rows we sent that are NOT in the
table — which would be the actually alarming case.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import breakage_rows, connect, load_frames, release_rows  # noqa: E402


def audit_other_tables(engine, changes, rels) -> None:
    """A fixture breakage row implies fixture rows behind it.

    A breakage cannot exist without a release, and a release cannot exist
    without a package, so six hand-written breakages guarantee at least
    one hand-written release. That release is the one that matters: it is
    what makes the site say "pandas 2.2.0 removed DataFrame.append".

    model_run is checked for a different reason. `--scores` writes a new
    model_run and points every prediction at it, and the API picks a run
    to display. If a fixture run is sitting there with invented metrics,
    it is competing with a real one for that slot — and a fake PR-AUC on
    the site is the single most embarrassing thing this project could
    ship. Better to know before writing, not after.
    """
    from sqlalchemy import text

    ours_rel = {(r["package"], r["version"]) for r in rels}
    ours_pkg = set(changes["package"])

    with engine.connect() as conn:
        db_rel = conn.execute(text("""
            SELECT p.name, r.version, r.released_at, r.bump_type
            FROM release r JOIN package p ON p.id = r.package_id
        """)).fetchall()
        runs = conn.execute(text("""
            SELECT version, pr_auc, precision_at_10, ndcg_at_20
            FROM model_run ORDER BY version
        """)).fetchall()
        n_pred = conn.execute(
            text("SELECT count(*) FROM prediction")).scalar()

    print("\n" + "=" * 68)
    print("  THE TABLES BEHIND THEM")
    print("=" * 68)

    extra_rel = [r for r in db_rel if (r[0], r[1]) not in ours_rel]
    print(f"release rows not from changes.csv: {len(extra_rel)}")
    for name, ver, at, bump in extra_rel[:15]:
        flag = "  <- package we never analysed" if name not in ours_pkg else ""
        print(f"   {name} {ver}   released_at={at}  bump={bump}{flag}")

    print(f"\nmodel_run rows: {len(runs)}")
    for version, pr, p10, ndcg in runs:
        print(f"   {version}   pr_auc={pr}  p@10={p10}  ndcg@20={ndcg}")
    if runs:
        print("   Check these against artifacts/metrics.json. A run whose")
        print("   numbers you do not recognise is a fixture, and --scores")
        print("   will add a real one alongside it rather than replacing it.")

    print(f"\nprediction rows: {n_pred:,}")


def main() -> None:
    changes, _usage, _packages = load_frames()

    # Same trick the dry run uses: a DISTINCT fake id per release, so that
    # breakage_rows deduplicates exactly as it does against the real ids.
    # The ids themselves are irrelevant here; only `keys` is used, and
    # those carry (package, version, symbol, kind, sub_target) — the same
    # tuple the loader reads back out of the database.
    rels = release_rows(changes)
    fake = {(r["package"], r["version"]): i for i, r in enumerate(rels, 1)}
    _rows, keys, dropped = breakage_rows(changes, fake)
    ours = set(keys)
    print(f"built from changes.csv: {len(ours):,} distinct keys "
          f"({dropped} dropped as duplicates)")

    from sqlalchemy import text

    engine = connect()
    with engine.connect() as conn:
        has_sub = bool(conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'breakage' AND column_name = 'sub_target'
        """)).fetchone())
        sub_sel = "b.sub_target" if has_sub else "''"

        found = conn.execute(text(f"""
            SELECT b.id, p.name, r.version, b.symbol_path, b.kind, {sub_sel},
                   b.detail, b.is_private
            FROM breakage b
            JOIN release r ON r.id = b.release_id
            JOIN package p ON p.id = r.package_id
            ORDER BY b.id
        """)).fetchall()

        orphans = conn.execute(text("""
            SELECT count(*) FROM breakage b
            LEFT JOIN release r ON r.id = b.release_id
            WHERE r.id IS NULL
        """)).scalar()

    print(f"in the breakage table:  {len(found):,} rows "
          f"(sub_target column present: {has_sub})")
    if orphans:
        print(f"\n** {orphans:,} breakage rows point at a release_id that "
              "does not exist.\n** Those cannot be joined and would never "
              "appear on the site.")

    in_db = {(name, ver, sym, kind, sub or ""): (bid, detail, priv)
             for bid, name, ver, sym, kind, sub, detail, priv in found}

    extra = sorted(k for k in in_db if k not in ours)
    missing = sorted(k for k in ours if k not in in_db)

    print(f"\nrows in the table that we did NOT send: {len(extra)}")
    print(f"rows we sent that are NOT in the table: {len(missing)}")

    if missing:
        print("\n** This is the direction that matters. A row we sent and")
        print("** cannot read back means the insert did not do what it")
        print("** claims. Do not run --scores until this is understood.")
        for k in missing[:10]:
            print("   ", k)

    # Run this whether or not there are extra breakage rows: a fixture
    # release or model_run can exist without any breakage pointing at it.
    audit_other_tables(engine, changes, rels)

    if not extra:
        print("\nNothing unexplained in breakage. The table contains exactly "
              "what the pipeline put there.")
        return

    print("\n" + "=" * 68)
    print("  THE ROWS WE DID NOT SEND")
    print("=" * 68)
    # Print a sample, not all of them. This started life expecting 6 rows
    # and printed every one in full; at 223 that is several screens of
    # JSON nobody reads, and the classification below is the part that
    # actually answers the question.
    SHOW = 12
    if len(extra) > SHOW:
        print(f"showing {SHOW} of {len(extra)} — the breakdown below is the "
              "part that matters\n")
    for k in extra[:SHOW]:
        bid, detail, priv = in_db[k]
        pkg, ver, sym, kind, sub = k
        print(f"\nid {bid}   {pkg} {ver}")
        print(f"  symbol   {sym}")
        print(f"  kind     {kind}"
              + (f"   sub_target {sub!r}" if sub else ""))
        print(f"  private  {priv}")
        try:
            d = json.loads(detail) if isinstance(detail, str) else detail
            print(f"  detail   {json.dumps(d)[:300]}")
        except (TypeError, ValueError):
            print(f"  detail   {str(detail)[:300]}")

    pkgs = {k[0] for k in extra}
    print("\n" + "-" * 68)
    print(f"They span {len(pkgs)} package(s): {', '.join(sorted(pkgs))}")
    if len(pkgs) == 1:
        print("All in one package — consistent with hand-written fixtures "
              "rather\nthan a pipeline that half-ran.")

    # CLASSIFY. Rows we did not send are not all the same thing, and an
    # earlier version of this script said any package missing from
    # changes.csv was "strong evidence these were inserted by hand". That
    # is wrong and it pointed at the wrong conclusion: grpcio, brotli,
    # safetensors and backports-zstd were never hand-inserted — they
    # SUCCEEDED on an earlier run and failed on this one. The loader has
    # no DELETE, so the table is the union of every run ever made.
    #
    # Three categories, three different correct actions:
    real_pkgs = set(changes["package"])
    live_rel = set(zip(changes["package"], changes["version_to"])) | \
               set(zip(changes["package"], changes["version_from"]))

    # 1. Un-importable roots. `python.grpcio.grpc.StatusCode.OK` and
    #    `bindings.python.py_src.tokenizers...` are directory names the
    #    module detector walked into (NOTES 9.5). Nobody can import them,
    #    so they are wrong regardless of which run produced them, and the
    #    real fix is at ingest.
    BAD_ROOTS = ("python.", "bindings.", "py_src.", "crates.", "_cffi_src.")
    bad = [k for k in extra if k[2].startswith(BAD_ROOTS)]

    # 2. The release is still in the current dataset but this row is not.
    #    We re-analysed that exact release and did not produce this row,
    #    so it is superseded — the safest thing here is deletion.
    superseded = [k for k in extra
                  if k not in bad and (k[0], k[1]) in live_rel]

    # 3. The release has left the six-release window. These are TRUE
    #    findings about real releases and deleting them makes the product
    #    worse: someone upgrading litellm 1.95 -> 1.97 still wants to know
    #    what 1.96 broke. Accumulation is the correct behaviour here.
    aged_out = [k for k in extra if k not in bad and k not in superseded]

    print("\n" + "-" * 68)
    print("WHAT KIND OF EXTRA ARE THEY?")
    print(f"  un-importable symbol root (NOTES 9.5):  {len(bad):>4}"
          "   <- delete, these are wrong")
    print(f"  release re-analysed, row not produced:  {len(superseded):>4}"
          "   <- delete, superseded")
    print(f"  release aged out of the window:         {len(aged_out):>4}"
          "   <- KEEP, still true")
    for name, group in (("un-importable", bad), ("superseded", superseded),
                        ("aged out", aged_out)):
        if group:
            ps = sorted({k[0] for k in group})
            print(f"\n  {name}: {', '.join(ps[:10])}"
                  + (" ..." if len(ps) > 10 else ""))

    print("\n" + "-" * 68)
    print("NOT RUN — this is a suggestion, not an action. These rows are in")
    print("a table Varad also writes to, so agree with him before deleting")
    print("anything. If you both decide they are fixtures:\n")
    removable = bad + superseded
    if removable:
        ids = ", ".join(str(in_db[k][0]) for k in removable)
        print(f"  DELETE FROM prediction WHERE breakage_id IN ({ids});")
        print(f"  DELETE FROM breakage   WHERE id IN ({ids});")
        print(f"\n  That is {len(removable)} rows — the un-importable and the")
        print(f"  superseded. The {len(aged_out)} aged-out rows are NOT in it.")
    else:
        print("  (nothing safe to delete automatically)")
    print("\nCheck the release and package tables afterwards too — a fixture")
    print("breakage usually arrives with a fixture release behind it.")


if __name__ == "__main__":
    main()
