"""
The offline bundle (NOTES §38.7): one SQLite file holding every analysed
change and every release the loader establishes a status for, built by
scripts/build_bundle.py from the pipeline's CSV files through the loader's
own functions, so it counts what the site counts. This module only READS
it; the standard library's sqlite3 is enough, and nothing here touches the
network.

Three tables and a manifest:

  package   name, download_rank
  release   package, version, released_at, analysis_status, n_changes
  change    package, version_from, version_to, symbol, export_paths, kind,
            sub_target, message (griffe's words, the file location
            stripped as the loader strips it), old_value, new_value (where
            the message carries both), is_private, module_depth,
            public_depth, inherited_by, user_count, score (path length,
            the ranker that ships, §36)
  manifest  key, value: built_at, the source files and their hashes, the
            row counts, the griffe version, the schema version

Version order is packaging's, as the loader's (_version_order): a version
that does not parse sorts after every one that does, by its text.

The reader answers four questions and nothing else: what does the bundle
hold for a package, which releases lie in an upgrade (from, to], what
changed in them, and when was the bundle built. A package or version it
does not hold is reported, never guessed (§38.2).
"""

from __future__ import annotations

import ast
import dataclasses
import os
import pathlib
import sqlite3

from packaging.version import InvalidVersion, Version

SCHEMA_VERSION = "1"
DEFAULT_PATH = (pathlib.Path(__file__).resolve().parent / "data"
                / "breakrank.sqlite")
ENV_VAR = "BREAKRANK_BUNDLE"

# Which statuses are a verdict on the pair (previous -> V), and which say
# only that the run did not compare V (NOTES §33.9). The same two sets as
# ml/db.py; spelled here so the installed tool needs none of ml/.
COMPARED = ("analysed", "analysed_clean")
NOT_COMPARED = ("no_baseline", "analysis_failed", "no_source")

SCHEMA = """
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE package (name TEXT PRIMARY KEY, download_rank INTEGER);
CREATE TABLE release (
    package TEXT NOT NULL, version TEXT NOT NULL, released_at TEXT,
    analysis_status TEXT NOT NULL, n_changes INTEGER NOT NULL,
    PRIMARY KEY (package, version));
CREATE TABLE change (
    id INTEGER PRIMARY KEY,
    package TEXT NOT NULL, version_from TEXT NOT NULL,
    version_to TEXT NOT NULL,
    symbol TEXT NOT NULL, export_paths TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL, sub_target TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL, old_value TEXT, new_value TEXT,
    is_private INTEGER NOT NULL, module_depth INTEGER NOT NULL,
    public_depth INTEGER NOT NULL, inherited_by INTEGER NOT NULL,
    user_count INTEGER NOT NULL, score REAL NOT NULL);
CREATE INDEX change_release ON change (package, version_to);
"""


def version_key(v: str):
    """Sort key: PEP 440 order where the text parses, text order after."""
    try:
        return (0, Version(str(v)), "")
    except InvalidVersion:
        return (1, Version("0"), str(v))


def _is_expression(text: str) -> bool:
    try:
        ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return False
    return True


def split_change(message: str) -> tuple[str, str] | None:
    """(old, new) out of griffe's "... was changed: OLD -> NEW", for a
    changed default or a changed attribute value, or None.

    griffe joins the two with " -> ", and a value can contain that text
    itself (sep=" -> " is a real default), so every split point is tried
    and one is kept only if both sides parse as Python expressions; exactly
    one must survive, else nothing is claimed. The loader's default_change
    does this for defaults alone; the bundle needs attribute values too.
    """
    marker = " was changed: "
    i = message.find(marker)
    if i < 0:
        return None
    rest = message[i + len(marker):].strip()
    parts = rest.split(" -> ")
    splits = [(" -> ".join(parts[:k]), " -> ".join(parts[k:]))
              for k in range(1, len(parts))]
    valid = [(a, b) for a, b in splits
             if _is_expression(a) and _is_expression(b)]
    return valid[0] if len(valid) == 1 else None


@dataclasses.dataclass(frozen=True)
class Release:
    package: str
    version: str
    released_at: str | None
    analysis_status: str
    n_changes: int

    @property
    def compared(self) -> bool:
        return self.analysis_status in COMPARED

    def describe(self) -> str:
        """One line, the site's wording for each status (NOTES §33.8)."""
        s = self.analysis_status
        if s == "analysed":
            return (f"compared with the one before it: {self.n_changes} "
                    "change(s)")
        if s == "analysed_clean":
            return "compared with the one before it: no breaking change"
        if s == "no_baseline":
            return "not compared: the analysis window starts at this release"
        if s == "analysis_failed":
            return "not compared: the analysis of this release failed"
        if s == "no_source":
            return "not compared: no source distribution"
        if s == "yanked":
            return "not compared: yanked on PyPI"
        return "we can't confirm it was compared with the one before it"


@dataclasses.dataclass(frozen=True)
class Change:
    id: int
    package: str
    version_from: str
    version_to: str
    symbol: str
    export_paths: tuple[str, ...]
    kind: str
    sub_target: str
    message: str
    old_value: str | None
    new_value: str | None
    is_private: bool
    module_depth: int
    public_depth: int
    inherited_by: int
    user_count: int
    score: float

    @property
    def paths(self) -> tuple[str, ...]:
        """Every dotted path downstream code can write for this symbol:
        the definition path first, then the export paths (§38.3)."""
        return (self.symbol,) + tuple(p for p in self.export_paths
                                      if p and p != self.symbol)


class BundleError(Exception):
    """The bundle cannot answer: missing file, unknown package or version."""


class Bundle:
    """A read-only view of one bundle file."""

    def __init__(self, path: pathlib.Path | str):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise BundleError(
                f"no bundle at {self.path}. Build one with "
                "python scripts/build_bundle.py, or point "
                f"{ENV_VAR} at one.")
        uri = f"file:{self.path.as_posix()}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = sqlite3.Row
        found = self.manifest().get("schema_version")
        if found != SCHEMA_VERSION:
            raise BundleError(
                f"bundle schema {found!r} at {self.path}; this tool reads "
                f"schema {SCHEMA_VERSION}. Rebuild it with "
                "python scripts/build_bundle.py.")

    @classmethod
    def default(cls) -> "Bundle":
        """The bundle inside the tool, or the one $BREAKRANK_BUNDLE names."""
        return cls(os.environ.get(ENV_VAR) or DEFAULT_PATH)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Bundle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ manifest

    def manifest(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in
                self.conn.execute("SELECT key, value FROM manifest")}

    def built_at(self) -> str:
        return self.manifest().get("built_at", "unknown")

    # ------------------------------------------------------------ packages

    def packages(self) -> list[str]:
        return [r["name"] for r in
                self.conn.execute("SELECT name FROM package ORDER BY name")]

    def has(self, package: str) -> bool:
        found = self.conn.execute(
            "SELECT 1 FROM package WHERE name = ?", (package,)).fetchone()
        return found is not None

    # ------------------------------------------------------------ releases

    def releases(self, package: str) -> list[Release]:
        """Every release the bundle holds for the package, oldest first."""
        if not self.has(package):
            raise BundleError(
                f"the bundle (built {self.built_at()}) has no package "
                f"named {package!r}; nothing is guessed about it.")
        rows = self.conn.execute(
            "SELECT package, version, released_at, analysis_status, n_changes "
            "FROM release WHERE package = ?", (package,)).fetchall()
        out = [Release(r["package"], r["version"], r["released_at"],
                       r["analysis_status"], int(r["n_changes"]))
               for r in rows]
        return sorted(out, key=lambda r: version_key(r.version))

    def newest(self, package: str) -> Release:
        return self.releases(package)[-1]

    def upgrade(self, package: str, frm: str, to: str) -> list[Release]:
        """The releases an upgrade crosses: every one in (frm, to], oldest
        first, as the site's /analyze walks a range (NOTES §8). `frm` and
        `to` must both be releases the bundle holds; an empty range is
        reported, not returned as a silent nothing."""
        rels = self.releases(package)
        have = {r.version for r in rels}
        for v, word in ((frm, "from"), (to, "to")):
            if v not in have:
                newest = rels[-1].version
                raise BundleError(
                    f"{package} {v} is not a release the bundle holds "
                    f"(--{word}). The bundle was built {self.built_at()} "
                    f"and its newest {package} is {newest}. A release it "
                    "does not hold needs the ingest, which fetches the "
                    "source distributions over the network; the tool does "
                    "not guess.")
        lo, hi = version_key(frm), version_key(to)
        if not lo < hi:
            raise BundleError(
                f"nothing to upgrade: {package} {frm} is not older than "
                f"{to}." + (" The new version is already installed."
                             if lo == hi else ""))
        return [r for r in rels if lo < version_key(r.version) <= hi]

    # ------------------------------------------------------------- changes

    def changes(self, package: str, versions: list[str] | None = None
                ) -> list[Change]:
        """The changes of the given releases (every version_to in
        `versions`; all of the package's when None), in the site's order:
        by score, highest first, then symbol, then kind and sub_target."""
        sql = ("SELECT * FROM change WHERE package = ?")
        args: list = [package]
        if versions is not None:
            if not versions:
                return []
            sql += " AND version_to IN (%s)" % ",".join("?" * len(versions))
            args += list(versions)
        rows = self.conn.execute(sql, args).fetchall()
        out = [Change(
            int(r["id"]), r["package"], r["version_from"], r["version_to"],
            r["symbol"],
            tuple(p for p in (r["export_paths"] or "").split(";") if p),
            r["kind"], r["sub_target"] or "", r["message"],
            r["old_value"], r["new_value"], bool(r["is_private"]),
            int(r["module_depth"]), int(r["public_depth"]),
            int(r["inherited_by"]), int(r["user_count"]), float(r["score"]))
            for r in rows]
        return sorted(out, key=lambda c: (-c.score, c.symbol, c.kind,
                                          c.sub_target))

    def counts(self) -> dict[str, int]:
        q = self.conn.execute
        return {"packages": q("SELECT count(*) FROM package").fetchone()[0],
                "releases": q("SELECT count(*) FROM release").fetchone()[0],
                "changes": q("SELECT count(*) FROM change").fetchone()[0]}
