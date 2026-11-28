"""
The project's environment, read from disk and never imported (NOTES §38.2):
which version of a distribution is installed, and which import roots it
owns, from its `*.dist-info` directory.

The tool is run with the project's own interpreter (`python -m fixer`), so
the default environment is that interpreter's site-packages, from
sysconfig. `--env PATH` names another: a venv root (`.venv`), or a
site-packages directory itself. Two environments are two runs.

Distribution names and import names differ (the ingest's `resolve_layout`
has the same job for sdists): `databricks-sdk` ships `databricks`, and the
`griffe` distribution is a shell whose code is in `griffelib`. The
dist-info directory is named by the NORMALISED distribution name
(PEP 503: lower case, runs of `-`, `_` and `.` as one `_`), and its
`top_level.txt`, or failing that its `RECORD`, says which top-level
packages it installed. Nothing here runs any code of the environment.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
import re
import sysconfig


def normalise(name: str) -> str:
    """PEP 503's normalisation, then `_` as the dist-info spelling."""
    return re.sub(r"[-_.]+", "-", name).lower().replace("-", "_")


class EnvError(Exception):
    """The environment cannot be read, or does not hold the distribution."""


@dataclasses.dataclass(frozen=True)
class Installed:
    distribution: str          # the name METADATA gives
    version: str
    import_roots: tuple[str, ...]
    dist_info: pathlib.Path


def site_packages(env: str | pathlib.Path | None) -> pathlib.Path:
    """The directory holding the `*.dist-info` entries: the given one when
    it holds any (a site-packages or dist-packages directory), the one
    inside a venv root, or the running interpreter's."""
    if env is None:
        return pathlib.Path(sysconfig.get_paths()["purelib"])
    p = pathlib.Path(env).expanduser().resolve()
    if not p.is_dir():
        raise EnvError(f"--env {p} is not a directory")
    if any(p.glob("*.dist-info")):
        return p
    found = sorted(p.glob("lib/python3.*/site-packages")) \
        + sorted(p.glob("Lib/site-packages"))
    if found:
        return found[0]
    raise EnvError(f"--env {p} is neither a directory of installed "
                   "distributions (no *.dist-info in it) nor a venv root "
                   "(no lib/python3.*/site-packages inside it)")


def _metadata(dist_info: pathlib.Path) -> dict[str, str]:
    """The first value of each header in METADATA (name, version are the
    two read); no imports, no parsing of the body."""
    out: dict[str, str] = {}
    try:
        text = (dist_info / "METADATA").read_text(encoding="utf-8",
                                                  errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, _, value = line.partition(":")
            out.setdefault(key.strip().lower(), value.strip())
    return out


def _roots(dist_info: pathlib.Path) -> tuple[str, ...]:
    """top_level.txt when present, else the first path component of every
    file RECORD lists inside the directory: packages and `.py` modules,
    not the dist-info itself, caches, scripts outside it (`../bin/x`) or
    compiled extension modules, which griffe cannot read statically."""
    top = dist_info / "top_level.txt"
    if top.exists():
        names = [ln.strip() for ln in
                 top.read_text(encoding="utf-8",
                               errors="replace").splitlines()]
        roots = tuple(sorted({n for n in names if n}))
        if roots:
            return roots
    record = dist_info / "RECORD"
    roots: set[str] = set()
    if record.exists():
        with record.open(encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                first = row[0].replace("\\", "/").split("/")[0]
                if (not first or first in (".", "..", "__pycache__")
                        or first.endswith((".dist-info", ".pth", ".so",
                                           ".pyd", ".dylib", ".txt"))):
                    continue
                roots.add(first[:-3] if first.endswith(".py") else first)
    return tuple(sorted(roots))


def find_installed(package: str, env: str | pathlib.Path | None = None
                   ) -> Installed:
    """The installed distribution `package`, or EnvError saying which
    environment was read and what it lacks."""
    site = site_packages(env)
    want = normalise(package)
    candidates = []
    for d in site.glob("*.dist-info"):
        stem = d.name[:-len(".dist-info")]
        dist_part = stem.rsplit("-", 1)[0] if "-" in stem else stem
        if normalise(dist_part) != want:
            continue
        meta = _metadata(d)
        if normalise(meta.get("name", dist_part)) != want:
            continue
        version = meta.get("version") or (
            stem.rsplit("-", 1)[1] if "-" in stem else "")
        candidates.append(Installed(meta.get("name", dist_part), version,
                                    _roots(d), d))
    if not candidates:
        raise EnvError(f"{package} is not installed in {site} (no "
                       f"{want}-*.dist-info there). Give --from, or --env "
                       "for the environment that has it.")
    if len(candidates) > 1:
        versions = ", ".join(sorted(c.version for c in candidates))
        raise EnvError(f"{package} is installed more than once in {site} "
                       f"({versions}); give --from.")
    return candidates[0]
