"""
Remove the version-string and package-metadata rows from the breakage
table. Nothing else.

    python scripts/db_drop_versions.py                 # dry run: SELECTs only
    python scripts/db_drop_versions.py --expect 2187   # delete, if the count
                                                       # is still 2187

WHAT GOES (NOTES §33.5). Every breakage row that build.py's own two rules
mark, version_strings() and metadata_strings(), read from the row's
symbol_path, kind and detail.griffe_message: a version constant
(pkg.__version__, a module's `version`) or a package-metadata string
(pkg.__copyright__, pkg.__author__) whose VALUE changed. Such a row
describes the release; it does not change it, and it breaks nobody. F1
(§23, narrowed in §33.1) and F39 (§32) took them out of the model, and
from 12 Oct the loader does not write them (ml/db.py, not_a_change). This
removes the ones earlier loads wrote, whichever load it was: the rule is
about what a row is, not which run produced it. A removed version
constant, or one changed to `unset`, is a change, and stays.

Not the rows of the package `breakrank-fixture`, which are Varad's
(§8.1): they are counted and left alone.

WHAT GOES WITH THEM, in the same transaction:
  - their prediction rows, which are earlier models' scores: the ranker
    that ships and the fallback score none of these rows (§33.4, §36.4);
  - each release that lost rows has n_changes recounted from the table,
    and one left with no row and reading `analysed` becomes
    analysed_clean, which is what the loader writes for it now: it was
    diffed, and what the diff found was its version string.

WHY --expect, AND NOT --yes. The number Varad reads in the dry run must
be the number that goes. So this deletes only when told the count it
should find, and refuses before the transaction opens if the table has
moved: a load in between, a different database, a changed rule.

WHY NOT db_prune.py. db_prune refuses to delete more than 5% of the table,
on purpose (a stale changes.csv and a wrong database produce the same
large number). These rows are 6.5% of it by the loader's count (§33.2),
and they are not "superseded" or "un-importable" rows but a kind of their
own, so they get a script of their own, and db_prune's guard stands.
db_prune counts them as a category it never deletes.

THE ORDER ON NEON (§33.5): this dry run; its output to Varad; a Neon
branch made as the restore point; this script with --expect; then
`python ml/db.py` (no --scores: the scores stand), which writes the
releases the files cannot establish as `unknown`; then db_prune's dry run.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.db import FIXTURE_PACKAGE, connect  # noqa: E402
from ml.features.build import metadata_strings, version_strings  # noqa: E402

COLUMNS = ["id", "package", "version", "symbol", "kind", "explanation",
           "release_id", "status", "n_changes"]
RULES = ("version string", "package metadata")


def marked(rows: pd.DataFrame) -> pd.DataFrame:
    """The table's rows with two columns added: `rule`, which of the two
    rules marks the row ("" for neither), and `fixture`, whether it is
    Varad's. build.py's functions read `symbol`, `kind` and `explanation`,
    the names changes.csv uses, so the table's columns are given those
    names in the SELECT below. A version string that is also a metadata
    name (there is none today) counts once, as a version string."""
    out = rows.copy()
    if not len(out):
        out["rule"] = pd.Series(dtype=str)
        out["fixture"] = pd.Series(dtype=bool)
        return out
    out["explanation"] = out["explanation"].fillna("").astype(str)
    vs = version_strings(out)
    meta = metadata_strings(out) & ~vs
    out["rule"] = ""
    out.loc[meta, "rule"] = RULES[1]
    out.loc[vs, "rule"] = RULES[0]
    out["fixture"] = out["package"].astype(str).eq(FIXTURE_PACKAGE)
    return out


def plan(rows: pd.DataFrame) -> dict:
    """What a deletion would do, from the marked rows. Pure, so the test
    can check it; main() prints it and, with --expect, carries it out."""
    m = marked(rows)
    go = m[(m["rule"] != "") & ~m["fixture"]]
    skipped = m[(m["rule"] != "") & m["fixture"]]
    per_release = go.groupby("release_id").size()
    left = (m.groupby("release_id").size() - per_release).reindex(
        per_release.index).fillna(0).astype(int)
    status = m.drop_duplicates("release_id").set_index("release_id")["status"]
    emptied = left[left == 0].index
    to_clean = [int(r) for r in emptied
                if status.get(r) == "analysed"]
    leaves = (go["symbol"].astype(str).str.rsplit(".", n=1).str[-1]
              .value_counts())
    return {
        "total": int(len(m)),
        "doomed": [int(i) for i in go["id"]],
        "by_rule": {r: int((go["rule"] == r).sum()) for r in RULES},
        "by_leaf": {k: int(v) for k, v in leaves.items()},
        "fixture_skipped": int(len(skipped)),
        "releases": [int(r) for r in per_release.index],
        "left": {int(r): int(n) for r, n in left.items()},
        "emptied": [int(r) for r in emptied],
        "to_clean": to_clean,
        "sample": [(str(p), str(v), str(s)) for p, v, s in
                   go.sort_values(["package", "version", "symbol"])
                   [["package", "version", "symbol"]].head(6).itertuples(
                       index=False)],
    }


def describe(p: dict, predictions: list) -> str:
    n = len(p["doomed"])
    share = n / max(p["total"], 1)
    lines = [f"\nbreakage rows in table: {p['total']:,}",
             f"  marked by build.py's rules, not {FIXTURE_PACKAGE}'s: {n:,}  "
             f"({share:.1%} of the table)"]
    for r in RULES:
        lines.append(f"    {r:<18}{p['by_rule'][r]:>7,}")
    if p["by_leaf"]:
        lines.append("  by name: " + "   ".join(
            f"{k} {v:,}" for k, v in list(p["by_leaf"].items())[:8])
            + ("   ..." if len(p["by_leaf"]) > 8 else ""))
    if p["sample"]:
        lines.append("  e.g. " + "; ".join(f"{a} {b}: {c}"
                                           for a, b, c in p["sample"]))
    lines.append(f"  {FIXTURE_PACKAGE} rows the rules mark, left alone: "
                 f"{p['fixture_skipped']:,}")
    lines.append(f"  releases that lose rows: {len(p['releases']):,}; left "
                 f"with none: {len(p['emptied']):,}, of which "
                 f"{len(p['to_clean']):,} read 'analysed' and become "
                 "analysed_clean")
    if predictions:
        lines.append("  prediction rows on them, by model version: " +
                     "   ".join(f"{v} {c:,}" for v, c in predictions))
    else:
        lines.append("  prediction rows on them: none")
    return "\n".join(lines)


def delete(conn, p: dict, has_status: bool) -> dict:
    """The one transaction: predictions, then the rows, then the
    releases' counts and the clean status. Returns what each step did."""
    from sqlalchemy import text

    ids, rids = p["doomed"], p["releases"]
    done = {"prediction": 0, "breakage": 0, "recounted": 0, "clean": 0,
            "left_marked": None}
    done["prediction"] = conn.execute(
        text("DELETE FROM prediction WHERE breakage_id = ANY(:ids)"),
        {"ids": ids}).rowcount
    done["breakage"] = conn.execute(
        text("DELETE FROM breakage WHERE id = ANY(:ids)"),
        {"ids": ids}).rowcount
    if has_status and rids:
        done["recounted"] = conn.execute(text("""
            UPDATE release SET n_changes = (SELECT count(*) FROM breakage b
                                            WHERE b.release_id = release.id)
            WHERE id = ANY(:rids)"""), {"rids": rids}).rowcount
        done["clean"] = conn.execute(text("""
            UPDATE release SET analysis_status = 'analysed_clean'
            WHERE id = ANY(:rids) AND n_changes = 0
              AND analysis_status = 'analysed'"""), {"rids": rids}).rowcount
    return done


def fetch(conn) -> tuple[pd.DataFrame, bool]:
    """Every breakage row with what the two rules read, and whether the
    release table has the status columns (migration 005)."""
    from sqlalchemy import text

    has_status = bool(conn.execute(text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'release' AND column_name = 'analysis_status'
    """)).fetchone())
    status_cols = ("r.analysis_status, r.n_changes" if has_status
                   else "NULL, NULL")
    found = conn.execute(text(f"""
        SELECT b.id, p.name, r.version, b.symbol_path, b.kind,
               b.detail ->> 'griffe_message', b.release_id, {status_cols}
        FROM breakage b
        JOIN release r ON r.id = b.release_id
        JOIN package p ON p.id = r.package_id
    """)).fetchall()
    return pd.DataFrame([tuple(r) for r in found], columns=COLUMNS), has_status


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete the version-string and package-metadata rows "
                    "(F1, F39) from the breakage table. Dry run by default.")
    ap.add_argument("--expect", type=int, default=None,
                    help="delete, if and only if this is the count the dry "
                         "run prints now. Without it nothing is written.")
    args = ap.parse_args()

    from sqlalchemy import text

    engine = connect()
    with engine.connect() as conn:
        rows, has_status = fetch(conn)
        p = plan(rows)
        predictions = conn.execute(text("""
            SELECT model_version, count(*) FROM prediction
            WHERE breakage_id = ANY(:ids) GROUP BY 1 ORDER BY 1
        """), {"ids": p["doomed"]}).fetchall() if p["doomed"] else []
    print(describe(p, [(v, int(c)) for v, c in predictions]))
    if not has_status:
        print("  (no analysis_status column: migration 005 is not applied, "
              "so no release is recounted)")

    n = len(p["doomed"])
    if not n:
        print("\nNothing to do: the table holds no such row.")
        return
    if args.expect is None:
        print(f"\nDry run. Nothing was deleted. To delete exactly these "
              f"{n:,} rows:\n  python scripts/db_drop_versions.py --expect "
              f"{n}\nFirst: send this output to Varad, and make a Neon "
              "branch as the restore point (NOTES §33.5).")
        return
    if args.expect != n:
        sys.exit(f"\nREFUSING: you expected {args.expect:,} rows and the "
                 f"table has {n:,} now.\nNothing was deleted. Read the dry "
                 "run above; if it is right, re-run with --expect "
                 f"{n}.")

    with engine.begin() as conn:
        done = delete(conn, p, has_status)
    print(f"\ndeleted {done['prediction']:,} prediction rows and "
          f"{done['breakage']:,} breakage rows, in one transaction;")
    print(f"recounted n_changes on {done['recounted']:,} releases, "
          f"{done['clean']:,} of them now analysed_clean.")

    with engine.connect() as conn:
        rows, _ = fetch(conn)
        after = plan(rows)
    print(f"rows the rules mark, after: {len(after['doomed']):,} (should be "
          "0; scripts/db_model_check.py counts the same).")
    print("Next: python ml/db.py (no --scores: the scores stand), then "
          "python scripts/db_prune.py (dry run).")


if __name__ == "__main__":
    main()
