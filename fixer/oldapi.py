"""
The old version's API, read STATICALLY from the environment with griffe
(NOTES §38.3, §38.4): which class defines a method a subclass is called
with, and what a function's parameters are, in order, with their kinds and
defaults. R1 needs the second to map a position to a name; locate needs the
first so that `Redis(...).set(...)` matches a change recorded on the mixin
that defines `set`.

griffe is the ingest's extractor, and it is used here the way the ingest
uses it: `allow_inspection=False`, so a module griffe cannot parse is left
unread rather than imported (§17; the ingest calls that flag a security
boundary, not a tuning knob). A package griffe cannot find in the
environment (an editable install behind a `.pth` file, a compiled
module, a distribution whose import roots are not where dist-info says)
gives an OldApi with nothing in it, and every question then answers None,
which the caller reports as "the old signature could not be read" and
flags instead of rewriting. A path griffe cannot resolve inside a loaded
package (one passing through an import of another package, which
`resolve_external=False` leaves unresolved, or a cyclic alias) answers
None too, never an exception. Nothing here guesses.
"""

from __future__ import annotations

import dataclasses
import pathlib

try:
    import griffe
except ImportError:  # pragma: no cover - the dependency is declared
    griffe = None


@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    # positional_only | positional_or_keyword | var_positional |
    # keyword_only | var_keyword
    kind: str
    default: str | None


class OldApi:
    """One package's old API, loaded once per run."""

    def __init__(self, roots: tuple[str, ...], site: pathlib.Path | None,
                 modules=None):
        self.roots = roots
        self.site = site
        self.modules: dict = {}
        self.errors: dict[str, str] = {}
        if modules is not None:            # a loaded collection, for tests
            self.modules = dict(modules)
            return
        if griffe is None or site is None:
            return
        for root in roots:
            try:
                self.modules[root] = griffe.load(
                    root, search_paths=[str(site)], allow_inspection=False,
                    resolve_aliases=True, resolve_external=False)
            except Exception as e:  # noqa: BLE001 - reported, never raised
                self.errors[root] = f"{type(e).__name__}: {e}"

    @property
    def loaded(self) -> bool:
        return bool(self.modules)

    # ---------------------------------------------------------------- lookup

    def lookup(self, path: str):
        """The griffe object at a dotted path, aliases followed to their
        target, or None (also for a path griffe cannot resolve: through an
        external package's import, or a cyclic alias)."""
        root, _, rest = path.partition(".")
        mod = self.modules.get(root)
        if mod is None:
            return None
        try:
            obj = mod[rest] if rest else mod
            if getattr(obj, "is_alias", False):
                obj = obj.final_target
        except Exception:  # noqa: BLE001 - unresolvable is None, not a crash
            return None
        return obj

    def canonical(self, path: str) -> str | None:
        """Where the object at `path` is defined: `redis.Redis` ->
        `redis.client.Redis`."""
        obj = self.lookup(path)
        return getattr(obj, "path", None) if obj is not None else None

    def mro(self, class_path: str) -> list[str]:
        """The defining paths of a class and its bases, nearest first;
        empty when the class is not known."""
        obj = self.lookup(class_path)
        if obj is None or not getattr(obj, "is_class", False):
            return []
        out = [obj.path]
        try:
            out += [k.path for k in obj.mro()]
        except Exception:  # noqa: BLE001 - a base griffe cannot resolve
            for base in getattr(obj, "bases", []):
                b = self.lookup(str(base))
                if b is not None and getattr(b, "is_class", False):
                    out.append(b.path)
        return out

    def defining_class(self, class_path: str, member: str) -> str | None:
        """Which class in `class_path`'s MRO defines `member`, by its
        defining path, or None when none does or the class is unknown."""
        for path in self.mro(class_path):
            cls = self.lookup(path)
            if cls is not None and member in getattr(cls, "members", {}):
                return path
        return None

    def labels(self, path: str) -> set[str]:
        """griffe's labels on the object: `classmethod`, `staticmethod`,
        `property`, ...; empty when unknown."""
        obj = self.lookup(path)
        try:
            return set(getattr(obj, "labels", ()) or ())
        except Exception:  # noqa: BLE001
            return set()

    def parameters(self, func_path: str) -> list[Param] | None:
        """A function's parameters in order (self included for a method,
        as griffe keeps them), or None when the function is not known or is
        not a function (a generated __init__ has nothing to read). A class
        answers with its `__init__`'s, its own or the inherited one."""
        obj = self.lookup(func_path)
        if obj is None:
            return None
        if getattr(obj, "is_class", False):
            try:
                members = obj.all_members
            except Exception:  # noqa: BLE001 - a base griffe cannot resolve
                members = getattr(obj, "members", {})
            obj = members.get("__init__") if "__init__" in members else None
            if obj is None:
                return None
            try:
                if getattr(obj, "is_alias", False):
                    obj = obj.final_target
            except Exception:  # noqa: BLE001
                return None
        if not getattr(obj, "is_function", False):
            return None
        out = []
        for p in obj.parameters:
            kind = getattr(p.kind, "name", str(p.kind))
            kind = kind.replace("ParameterKind.", "")
            out.append(Param(p.name, kind,
                             None if p.default is None else str(p.default)))
        return out
