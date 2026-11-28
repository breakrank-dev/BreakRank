"""Insert fixture data so API endpoints can be built and tested.

Lives under the package name 'breakrank-fixture', not 'pandas', for two
reasons. The real pipeline also loads pandas — versions 2.3.3, 3.0.0, 3.0.1,
3.0.2 — and fixture rows must never mix with real ones. And the fixture
versions are wrong on purpose-free grounds: DataFrame.append, iteritems and
read_csv(squeeze=) were all removed in pandas 2.0, not 2.2.0. On a site whose
whole claim is "we tell you which release broke you", a fixture leaking into a
demo must not make a false statement about a real package.

The symbol paths stay realistic (pandas.*) because they exercise the same code
paths in the explanation renderer. Only the package they hang from changed.

Safe to re-run. Delete before the demo.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.engine import engine

FIXTURE_PACKAGE = "breakrank-fixture"
FIXTURE_VERSION = "2.2.0"

# (symbol_path, kind, sub_target, is_private, user_count, score)
#
# The two read_csv rows are deliberate: two parameters removed from one
# function. Under the pre-004 unique key (release_id, symbol_path, kind) one
# of them silently vanished via ON CONFLICT DO NOTHING. They are a permanent
# regression case for migration 004.
#
# _parse_header is the private-symbol case for decision 3.
BREAKAGES = [
    ("pandas.DataFrame.append", "OBJECT_REMOVED", "", False, 412, 0.94),
    ("pandas.DataFrame.iteritems", "OBJECT_REMOVED", "", False, 88, 0.71),
    ("pandas.io.formats.style.Styler.where", "OBJECT_REMOVED", "", False, 12, 0.34),
    ("pandas.core.frame._parse_header", "OBJECT_REMOVED", "", True, 0, 0.02),
    ("pandas.read_csv", "PARAMETER_REMOVED", "verbose", False, 1180, 0.88),
    ("pandas.read_csv", "PARAMETER_REMOVED", "squeeze", False, 640, 0.79),
]


def main():
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO package (name, download_rank, github_repo)
                VALUES (:n, NULL, NULL)
                ON CONFLICT (name) DO NOTHING
            """),
            {"n": FIXTURE_PACKAGE},
        )

        pkg_id = conn.execute(
            text("SELECT id FROM package WHERE name = :n"),
            {"n": FIXTURE_PACKAGE},
        ).scalar()

        conn.execute(
            text("""
                INSERT INTO release
                    (package_id, version, released_at, bump_type,
                     analysis_status, n_changes)
                VALUES (:p, :v, :d, 'minor', 'analysed', :n)
                ON CONFLICT (package_id, version) DO UPDATE
                    SET analysis_status = EXCLUDED.analysis_status,
                        n_changes = EXCLUDED.n_changes
            """),
            {
                "p": pkg_id,
                "v": FIXTURE_VERSION,
                "d": datetime(2024, 1, 19, tzinfo=timezone.utc),
                "n": len(BREAKAGES),
            },
        )

        rel_id = conn.execute(
            text("SELECT id FROM release WHERE package_id = :p AND version = :v"),
            {"p": pkg_id, "v": FIXTURE_VERSION},
        ).scalar()

        # trained_at pinned to 2020 so this fixture can never win the
        # ORDER BY trained_at DESC model selection. Seeded with DEFAULT now()
        # it became the newest model run on every re-seed and silently beat
        # the real ranker.
        conn.execute(text("""
            INSERT INTO model_run
                (version, trained_at, pr_auc, precision_at_10, ndcg_at_20,
                 positive_rate, notes)
            VALUES ('v0-fake', '2020-01-01T00:00:00Z', 0.0, 0.0, 0.0, 0.0,
                    'Fixture. trained_at pinned to 2020 so it can never win '
                    'ORDER BY trained_at DESC model selection.')
            ON CONFLICT (version) DO UPDATE
                SET trained_at = EXCLUDED.trained_at,
                    notes = EXCLUDED.notes
        """))

        for symbol, kind, sub_target, private, users, score in BREAKAGES:
            conn.execute(
                text("""
                    INSERT INTO breakage
                        (release_id, symbol_path, kind, sub_target, is_private,
                         module_depth, is_top_level, in_dunder_all,
                         inherited_by, detail)
                    VALUES (:r, :s, :k, :sub, :priv, :depth, :top, TRUE, 0, :detail)
                    ON CONFLICT (release_id, symbol_path, kind, sub_target)
                        DO UPDATE SET detail = EXCLUDED.detail
                """),
                {
                    "r": rel_id,
                    "s": symbol,
                    "k": kind,
                    "sub": sub_target,
                    "priv": private,
                    "depth": symbol.count("."),
                    "top": symbol.count(".") <= 1,
                    "detail": json.dumps(
                        {"griffe_message": f"[fixture] {symbol} changed."}
                    ),
                },
            )

            b_id = conn.execute(
                text("""
                    SELECT id FROM breakage
                    WHERE release_id = :r AND symbol_path = :s
                      AND kind = :k AND sub_target = :sub
                """),
                {"r": rel_id, "s": symbol, "k": kind, "sub": sub_target},
            ).scalar()

            # usage_index is keyed on symbol only, so the two read_csv rows
            # share one entry and the second overwrites the first. That is
            # correct: usage is measured per symbol, not per parameter.
            conn.execute(
                text("""
                    INSERT INTO usage_index (symbol_path, user_count)
                    VALUES (:s, :u)
                    ON CONFLICT (symbol_path) DO UPDATE SET user_count = :u
                """),
                {"s": symbol, "u": users},
            )

            conn.execute(
                text("""
                    INSERT INTO prediction (breakage_id, model_version, score)
                    VALUES (:b, 'v0-fake', :sc)
                    ON CONFLICT (breakage_id, model_version) DO UPDATE SET score = :sc
                """),
                {"b": b_id, "sc": score},
            )

    print(
        f"Seeded {FIXTURE_PACKAGE} {FIXTURE_VERSION}: "
        f"{len(BREAKAGES)} breakages, model_run v0-fake (pinned 2020)."
    )


if __name__ == "__main__":
    main()