"""Apply SQL migrations in order. Safe to re-run."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.engine import engine

MIGRATIONS = Path(__file__).parent / "migrations"


def main():
    files = sorted(MIGRATIONS.glob("*.sql"))
    if not files:
        print("No migration files found.")
        return

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migration (
                filename   TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        applied = {
            r[0] for r in conn.execute(text("SELECT filename FROM schema_migration"))
        }

    for f in files:
        if f.name in applied:
            print(f"skip   {f.name}")
            continue

        sql = f.read_text().strip()
        if not sql:
            raise RuntimeError(
                f"{f.name} is empty. Refusing to record an empty migration "
                f"as applied - the file was probably never saved."
            )

        print(f"apply  {f.name}")
        with engine.begin() as conn:
            conn.execute(text(sql))
            conn.execute(
                text("INSERT INTO schema_migration (filename) VALUES (:n)"),
                {"n": f.name},
            )

    print("done")


if __name__ == "__main__":
    main()