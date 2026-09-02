"""Insert fake data so API endpoints can be built and tested.
NOT real data. Delete before the demo."""
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.db import engine

BREAKAGES = [
    ("pandas.DataFrame.append", "OBJECT_REMOVED", False, 412, 0.94),
    ("pandas.DataFrame.iteritems", "OBJECT_REMOVED", False, 88, 0.71),
    ("pandas.io.formats.style.Styler.where", "OBJECT_REMOVED", False, 12, 0.34),
    ("pandas.core.frame._parse_header", "OBJECT_REMOVED", True, 0, 0.02),
    ("pandas.read_csv", "PARAMETER_REMOVED", False, 1180, 0.88),
]


def main():
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO package (name, download_rank, github_repo)
            VALUES ('pandas', 12, 'pandas-dev/pandas')
            ON CONFLICT (name) DO NOTHING
        """))

        pkg_id = conn.execute(
            text("SELECT id FROM package WHERE name = 'pandas'")
        ).scalar()

        conn.execute(
            text("""
                INSERT INTO release (package_id, version, released_at, bump_type)
                VALUES (:p, '2.2.0', :d, 'minor')
                ON CONFLICT (package_id, version) DO NOTHING
            """),
            {"p": pkg_id, "d": datetime(2024, 1, 19, tzinfo=timezone.utc)},
        )

        rel_id = conn.execute(
            text("SELECT id FROM release WHERE package_id = :p AND version = '2.2.0'"),
            {"p": pkg_id},
        ).scalar()

        conn.execute(text("""
            INSERT INTO model_run (version, pr_auc, precision_at_10, ndcg_at_20, notes)
            VALUES ('v0-fake', 0.0, 0.0, 0.0, 'Placeholder. No model trained yet.')
            ON CONFLICT (version) DO NOTHING
        """))

        for symbol, kind, private, users, score in BREAKAGES:
            conn.execute(
                text("""
                    INSERT INTO breakage
                        (release_id, symbol_path, kind, is_private,
                         module_depth, is_top_level, in_dunder_all, explanation)
                    VALUES (:r, :s, :k, :priv, :depth, :top, TRUE, :expl)
                    ON CONFLICT (release_id, symbol_path, kind) DO NOTHING
                """),
                {
                    "r": rel_id, "s": symbol, "k": kind, "priv": private,
                    "depth": symbol.count("."),
                    "top": symbol.count(".") <= 1,
                    "expl": f"{symbol} was removed in this release.",
                },
            )

            b_id = conn.execute(
                text("SELECT id FROM breakage WHERE release_id = :r AND symbol_path = :s"),
                {"r": rel_id, "s": symbol},
            ).scalar()

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

    print("Seeded 1 package, 1 release, 5 breakages.")


if __name__ == "__main__":
    main()