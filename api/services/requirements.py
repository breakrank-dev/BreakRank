"""Turn a pasted requirements.txt into exact pins, and work out which releases
lie between each pin and the newest release we hold.

BreakRank can only say "this is what changes between the version you have and
the latest one" when it knows the version you have, so only exact pins (==)
are used. Every other line comes back with a reason; nothing is dropped
silently (decision 5).

This file has no database code, so it can be tested on its own.
"""

import re
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

# pip treats "#" as a comment when it starts the line or follows a space.
_COMMENT = re.compile(r"(^|\s)#.*$")

# Options pip allows after a requirement on the same line, such as
# --hash=sha256:... in files made by pip-compile --generate-hashes.
_TRAILING_OPTIONS = re.compile(r"\s+--\w.*$")

_MAX_SHOWN = 200  # characters of a skipped line to echo back


@dataclass(frozen=True)
class Pin:
    name: str  # canonical form: lower case, runs of - _ . become "-"
    version: Version
    line_number: int


@dataclass(frozen=True)
class SkippedLine:
    line_number: int
    line: str
    reason: str


def _logical_lines(text: str):
    """Yield (first line number, line), joining lines that end in a backslash, as pip does."""
    start, parts = None, []
    for number, raw in enumerate(text.splitlines(), start=1):
        if start is None:
            start = number
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            parts.append(stripped[:-1])
            continue
        parts.append(stripped)
        yield start, " ".join(parts)
        start, parts = None, []
    if parts:
        yield start, " ".join(parts)


def _why_not_pinned(req: Requirement) -> str | None:
    """None if the requirement pins one exact version, otherwise the reason it doesn't."""
    specs = list(req.specifier)
    if not specs:
        return "no version given (pin it with == to compare)"
    if len(specs) != 1 or specs[0].operator not in ("==", "===") or "*" in specs[0].version:
        return "not pinned to one exact version (use ==)"
    return None


def parse_requirements(text: str) -> tuple[list[Pin], list[SkippedLine]]:
    """Return the exact pins, plus every line we could not use and why."""
    pins: dict[str, Pin] = {}
    skipped: list[SkippedLine] = []

    for number, raw in _logical_lines(text):
        line = _COMMENT.sub("", raw).strip()
        if not line:
            continue  # blank line or comment only

        shown = raw.strip()[:_MAX_SHOWN]

        def skip(reason: str) -> None:
            skipped.append(SkippedLine(line_number=number, line=shown, reason=reason))

        if line.startswith("-"):
            skip("pip option, not a package")
            continue

        line = _TRAILING_OPTIONS.sub("", line)
        try:
            req = Requirement(line)
        except InvalidRequirement:
            skip("could not read this line as a requirement")
            continue

        if req.url:
            skip("installed from a URL or a path, not from PyPI")
            continue

        reason = _why_not_pinned(req)
        if reason:
            skip(reason)
            continue

        try:
            version = Version(list(req.specifier)[0].version)
        except InvalidVersion:
            skip("the pinned version is not a valid version number")
            continue

        name = canonicalize_name(req.name)
        if name in pins:
            skip(f"{name} is already pinned on line {pins[name].line_number}")
            continue

        pins[name] = Pin(name=name, version=version, line_number=number)

    return list(pins.values()), skipped


@dataclass(frozen=True)
class HeldRelease:
    id: int
    version: Version
    raw: str  # the version string exactly as stored
    analysis_status: str


def releases_after(current: Version, rows) -> tuple[HeldRelease | None, list[HeldRelease]]:
    """Given every release we hold for one package, return (latest, the ones after `current`).

    The list is oldest first. Pre-releases are left out unless you are already
    on one, the same rule pip follows. `rows` need .id, .version and
    .analysis_status attributes (database rows have them).
    """
    held = []
    for row in rows:
        try:
            version = Version(row.version)
        except InvalidVersion:
            continue
        if version.is_prerelease and not current.is_prerelease:
            continue
        held.append(HeldRelease(row.id, version, row.version, row.analysis_status))

    held.sort(key=lambda r: r.version)
    latest = held[-1] if held else None
    return latest, [r for r in held if r.version > current]
