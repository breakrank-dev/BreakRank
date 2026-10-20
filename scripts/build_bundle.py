"""
Build the fixer's offline bundle (NOTES §38.7) from the pipeline's CSV
files, through the loader's own functions, so it holds exactly the rows
the loader writes and the statuses the loader establishes.

    python scripts/build_bundle.py            # -> fixer/data/breakrank.sqlite
    python scripts/build_bundle.py --out some.sqlite

Reads  data/changes.csv, data/releases.csv, data/usage.csv, data/packages.csv
Writes one SQLite file (fixer/bundle.py describes the tables)

WHY THE FILES AND NOT THE DATABASE. The breakage table has no
export_paths column, and the short import path (`redis.Redis` for
`redis.client.Redis`) is what the fixer matches a project's imports
against (§38.3). So the bundle is today's dataset, the one the loader
writes (21,452 changes, 3,204 releases, 351 packages on 18 Oct), where
the site serves the union of every run; the difference is the aged-out
rows of earlier runs (§8), which have no changes.csv behind them.

THE SAME RULES AS THE LOADER, by calling the loader: package_rows for the
packages, release_plan for the releases and their statuses (a release
the files cannot establish is `unknown`, as the loader writes it since
migration 007), not_a_change for the rows that are not changes (§33.5),
and breakage_rows as the oracle for which rows survive the database key:
the records this script writes are checked against its rows and keys,
and the build stops if they differ. The score is path length's
(ml/model/ship.py, §36), computed within each release as the site's lists
are; public_depth as ml/features/labels.py computes it; user_count as the
site joins it, by the exact symbol.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import os
import pathlib
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fixer.bundle import (DEFAULT_PATH, SCHEMA, SCHEMA_VERSION,  # noqa: E402
                          split_change)
from ml.db import (CHANGES, FIXTURE_PACKAGE, GRIFFE_LOCATION,  # noqa: E402
                   PACKAGES, RELEASES, USAGE, UNKNOWN, _text,
                   breakage_rows, load_frames, load_releases, not_a_change,
                   package_rows, release_plan)
from ml.model.ship import path_scores  # noqa: E402

KEY = ["package", "version_to", "symbol", "kind", "sub_target"]


def public_depth(module_depth, export_paths) -> int:
    """Where the symbol can be reached: the shortest path a user can write
    (ml/features/labels.py)."""
    paths = [p for p in str(export_paths or "").split(";") if p]
    return int(min([int(module_depth)] + [p.count(".") for p in paths]))


def change_records(kept: pd.DataFrame, rels: dict, usage: pd.DataFrame
                   ) -> list[dict]:
    """One record per row the loader would write, in file order, deduped
    on the database key exactly as breakage_rows does (first wins)."""
    counts = (usage.drop_duplicates("symbol", keep="last")
              .set_index("symbol")["user_count"].to_dict())
    seen, out = set(), []
    for r in kept.itertuples(index=False):
        if (r.package, r.version_to) not in rels:
            continue
        sub = _text(r, "sub_target")
        key = (r.package, r.version_to, r.symbol, r.kind, sub)
        if key in seen:
            continue
        seen.add(key)
        message = GRIFFE_LOCATION.sub("", _text(r, "explanation"))
        values = (split_change(message)
                  if str(r.kind) in ("PARAMETER_CHANGED_DEFAULT",
                                     "ATTRIBUTE_CHANGED_VALUE") else None)
        exports = _text(r, "export_paths")
        out.append({
            "package": r.package, "version_from": str(r.version_from),
            "version_to": str(r.version_to), "symbol": r.symbol,
            "export_paths": exports, "kind": str(r.kind), "sub_target": sub,
            "message": message,
            "old_value": values[0] if values else None,
            "new_value": values[1] if values else None,
            "is_private": int(bool(r.is_private)),
            "module_depth": int(r.module_depth),
            "public_depth": public_depth(r.module_depth, exports),
            "inherited_by": int(getattr(r, "inherited_by", 0) or 0),
            "user_count": int(counts.get(r.symbol, 0)),
        })
    return out


def plan(changes, usage, packages, releases) -> dict:
    """Everything the bundle holds, as plain records, with the loader's
    rows as the check. Pure: no file is written here."""
    pkgs = [p for p in package_rows(changes, packages)
            if p["name"] != FIXTURE_PACKAGE]
    names = {p["name"] for p in pkgs}
    known, unknown, report = release_plan(changes, releases)
    for r in unknown:
        r["analysis_status"], r["n_changes"] = UNKNOWN, 0
    rels = [r for r in known + unknown if r["package"] in names]
    fake = {(r["package"], r["version"]): i for i, r in enumerate(rels, 1)}
    kept = changes[~not_a_change(changes)]
    kept = kept[kept["package"] != FIXTURE_PACKAGE]
    rows, keys, dropped = breakage_rows(kept, fake)
    records = change_records(kept, fake, usage)
    mine = [(c["package"], c["version_to"], c["symbol"], c["kind"],
             c["sub_target"]) for c in records]
    if len(records) != len(rows) or set(mine) != set(keys):
        sys.exit(f"REFUSING: this builder would write {len(records):,} "
                 f"changes and the loader writes {len(rows):,}; the two "
                 "must agree row for row. Nothing was written.")
    frame = pd.DataFrame(records)
    if len(frame):
        frame["score"] = path_scores(
            frame, within=["package", "version_from", "version_to"])
        records = frame.to_dict("records")
    return {"packages": pkgs, "releases": rels, "changes": records,
            "dropped": dropped, "not_changes": int(len(changes) - len(kept)),
            "report": report}


def sha256(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(out: pathlib.Path, p: dict, sources: dict) -> None:
    """One SQLite file, written beside `out` and moved into place whole."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO package VALUES (?, ?)",
                         [(x["name"], x["download_rank"])
                          for x in p["packages"]])
        conn.executemany(
            "INSERT INTO release VALUES (?, ?, ?, ?, ?)",
            [(r["package"], r["version"], r.get("released_at"),
              r["analysis_status"], int(r["n_changes"]))
             for r in p["releases"]])
        cols = ["package", "version_from", "version_to", "symbol",
                "export_paths", "kind", "sub_target", "message", "old_value",
                "new_value", "is_private", "module_depth", "public_depth",
                "inherited_by", "user_count", "score"]
        conn.executemany(
            f"INSERT INTO change ({', '.join(cols)}) VALUES "
            f"({', '.join('?' * len(cols))})",
            [tuple(c[k] for k in cols) for c in p["changes"]])
        try:
            griffe_version = importlib.metadata.version("griffe")
        except importlib.metadata.PackageNotFoundError:
            griffe_version = "unknown"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "built_at": dt.datetime.now(dt.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ranker": "path-length",
            "griffe_version": griffe_version,
            "n_packages": str(len(p["packages"])),
            "n_releases": str(len(p["releases"])),
            "n_changes": str(len(p["changes"])),
            "rows_not_changes": str(p["not_changes"]),
            "rows_dropped_duplicate_key": str(p["dropped"]),
        }
        for name, path in sources.items():
            manifest[f"source_{name}"] = str(path)
            manifest[f"sha256_{name}"] = sha256(path)
        conn.executemany("INSERT INTO manifest VALUES (?, ?)",
                         list(manifest.items()))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the fixer's offline bundle from the CSV files.")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_PATH,
                    help=f"where to write it (default {DEFAULT_PATH})")
    args = ap.parse_args()

    changes, usage, packages = load_frames()
    releases = load_releases()
    p = plan(changes, usage, packages, releases)
    write(args.out, p, {"changes": CHANGES, "releases": RELEASES,
                        "usage": USAGE, "packages": PACKAGES})
    statuses = pd.Series([r["analysis_status"] for r in p["releases"]]
                         ).value_counts()
    print(f"wrote {args.out}")
    print(f"  packages  {len(p['packages']):>7,}")
    print(f"  releases  {len(p['releases']):>7,}   "
          + "   ".join(f"{s} {n:,}" for s, n in statuses.items()))
    print(f"  changes   {len(p['changes']):>7,}   ({p['not_changes']:,} rows "
          "not changes, F1/F39, left out; "
          f"{p['dropped']:,} duplicate key(s) dropped, as the loader "
          "drops them)")
    if p["report"]["disagree"]:
        print(f"  releases.csv disagrees with changes.csv on "
              f"{len(p['report']['disagree'])} package(s); their statuses are "
              "not used, as in the loader")
    print("Check against the loader's last run (NOTES §33.10): 21,452 "
          "changes, 3,204 releases, 351 packages on the 18 Oct files.")


if __name__ == "__main__":
    main()
