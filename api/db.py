import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def psycopg_url(url: str) -> str:
    """Neon gives postgresql://...; ask SQLAlchemy for psycopg 3 instead of psycopg2.

    Only the prefix changes; the secret in .env and on Render stays as it is.
    """
    for prefix in ("postgresql+psycopg2://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


engine = create_engine(
    psycopg_url(os.environ["DATABASE_URL"]),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=2,
)