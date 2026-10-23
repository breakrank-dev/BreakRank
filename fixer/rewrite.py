"""
The rewrites (NOTES §38.4): R1, pass by name. R2, pin the old default,
follows in week 4; nothing else is rewritten.

A rewrite is decided per CALL SITE from every row of the called function
in the range, never per row: `Redis(...)` with 43 moved parameters is one
decision. R1 takes a resolved call (locate's `call` or `method` hit) and
the function's rows; where the call passes by position an argument whose
parameter was MOVED or became keyword-only, the positional arguments from
that one to the last are given their old names, `Client("h", 6380, 2)`
becoming `Client("h", port=6380, db=2)`; the names come from the OLD
signature, read statically with griffe (fixer/oldapi.py), never guessed.

R1 is withheld, with the reason printed, when:
  - the old signature could not be read (the old version not installed,
    or a signature griffe cannot see: a `*args, **kwargs` shell, a
    generated `__init__`);
  - the call uses `*` or `**` unpacking: the positions are unknown;
  - the call passes more positional arguments than the old signature has
    named ones: the rest went to `*args`, which has no name;
  - a parameter in the run was positional-only in the old version: it
    had no name to pass;
  - the function also carries a flag: a removed parameter, a new required
    one, a parameter made positional-only. Naming arguments after a
    removal mis-names every one after it, so the flag comes first, and
    the rewrite is printed under it as what it would be once the flag is
    dealt with;
  - the run would name a parameter the call also passes by keyword: the
    call is already wrong, and the tool does not guess which is meant.

A call that passes the moved parameters by name already, or does not pass
them, needs nothing, and is listed as such. griffe's positions count
`self`, so an instance call (`r.set(...)`), a constructor (`Redis(...)`)
and a classmethod called on the class (`Redis.from_url(...)`) are offset
by one and an explicit-self call (`Redis.set(r, ...)`) or a staticmethod
is not. libcst makes the edit and keeps the file's formatting, comments,
quotes, line endings and byte-order mark, so the diff shows the lines R1
changes and nothing else; a second run after `--apply` finds every call
it rewrote passing by name and rewrites nothing.

One premise the data cannot check (the review of 22 Oct): R1 assumes
that no parameter BEFORE the first moved one vanished without a row.
griffe reports no row for a parameter removed when the new signature
gains `*args` and `**kwargs` together (it counts as swallowed), and the
parameters after it as moved; `swallow(a, b, c)` -> `swallow(a, c, *args,
**kwargs)` is one MOVED row on `c`, and `swallow(1, 2, 3)` ->
`swallow(1, 2, c=3)` raises on the new version. The bundle's one-line
message carries no positions, so the tool cannot see it; §38.6's hand
check judges such a rewrite, and §38.11's ingest change (griffe's
`details`, with the positions) would let R1 withhold it.
"""

from __future__ import annotations

import dataclasses
import difflib
import pathlib
import re

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from fixer.bundle import Change
from fixer.locate import RESOLVED, Hit, Located
from fixer.oldapi import OldApi

REWRITE, FLAG, NOTHING = "rewrite", "flag", "nothing"

# The rows R1 acts on, and the rows that withhold it (§38.4).
MOVED_KINDS = {"PARAMETER_MOVED"}
BLOCKING_KINDS = {"PARAMETER_REMOVED", "PARAMETER_ADDED_REQUIRED"}
KIND_CHANGE = re.compile(r"Parameter kind was changed: (.+?) -> (.+?)\s*$")
NAMED = ("positional_only", "positional_or_keyword")


def kind_change(message: str) -> tuple[str, str] | None:
    """griffe's old and new parameter kinds out of a PARAMETER_CHANGED_KIND
    message (`... -> keyword-only`), or None."""
    m = KIND_CHANGE.search(message)
    return (m.group(1), m.group(2)) if m else None


@dataclasses.dataclass
class Plan:
    """R1's decision for one call site, from every parameter row of every
    function the call meets (one, unless an export path names two
    defining paths inside the range)."""
    hit: Hit
    # the function whose signature was read, by its defining path
    symbol: str
    action: str                      # rewrite | flag | nothing
    reason: str
    # every function with parameter rows the call meets; the decision
    # prints under each of their entries
    symbols: tuple[str, ...] = ()
    names: tuple[str, ...] = ()      # the parameters the run names
    start: int = 0                   # first positional argument renamed
    rows: tuple[int, ...] = ()       # the change ids behind the decision
    # the call's text after R1, filled from the source by `render`; for a
    # withheld rewrite, what it would be
    would_be: str | None = None

    @property
    def key(self) -> tuple:
        h = self.hit
        return (h.file, h.line, h.column, h.use, h.confidence)


def offset(hit: Hit, labels: set[str]) -> int:
    """How many leading parameters of the old signature the call does not
    pass: `self` for an instance call or a constructor, `cls` for a
    classmethod called on the class; none for a function, an explicit-self
    call or a staticmethod."""
    if "staticmethod" in labels:
        return 0
    if hit.use == "call":
        return 1 if hit.receiver else 0      # a constructor passes no self
    return 0 if hit.explicit_self else 1     # a method on an instance


def plan_r1(hit: Hit, symbol: str, rows: list[Change], params,
            labels: set[str], why_unread: str = "") -> Plan | None:
    """R1 for one resolved call of `symbol`, whose rows in the range are
    `rows` and whose old parameters are `params` (None when unreadable).
    None when R1 does not apply to this function at all."""
    moved: set[str] = set()
    flags: list[str] = []
    for c in rows:
        if c.kind in MOVED_KINDS:
            moved.add(c.sub_target)
        elif c.kind == "PARAMETER_CHANGED_KIND":
            kinds = kind_change(c.message)
            new = kinds[1] if kinds else ""
            if new == "keyword-only":
                moved.add(c.sub_target)
            elif new == "positional-only":
                flags.append(f"{c.sub_target} became positional-only in "
                             f"{c.version_to}")
            else:
                flags.append(f"{c.sub_target}'s kind changed"
                             + (f" ({kinds[0]} -> {kinds[1]})" if kinds
                                else "") + "; not a case R1 covers")
        elif c.kind in BLOCKING_KINDS:
            what = ("was removed" if c.kind == "PARAMETER_REMOVED"
                    else "was added as required")
            flags.append(f"{c.sub_target} {what} in {c.version_to}")
    if not moved:
        return None
    if hit.confidence != RESOLVED or hit.use not in ("call", "method"):
        return None
    ids = tuple(sorted(c.id for c in rows))
    plan = Plan(hit, symbol, NOTHING, "", rows=ids)
    if hit.star_args:
        return _flag(plan, "the call unpacks `*` or `**`: the positions "
                           "are unknown", flags)
    # Two decisions need no signature: every moved parameter is already
    # passed by name, or nothing is passed by position at all.
    flags.sort()
    if moved <= set(hit.keywords):
        plan.reason = "already passed by name: " + ", ".join(sorted(moved))
        if flags:
            plan.reason += "; flagged: " + "; ".join(flags)
        return plan
    if hit.positional == 0:
        plan.reason = ("the moved parameter(s) are not passed here: "
                       + _listed(moved))
        if flags:
            plan.reason += "; flagged: " + "; ".join(flags)
        return plan
    if params is None:
        return _flag(plan, "the old signature could not be read"
                           + (f": {why_unread}" if why_unread else ""),
                     flags)
    skip = offset(hit, labels)
    named = [p for p in params if p.kind in NAMED][skip:]
    if hit.positional > len(named):
        has_star = any(p.kind == "var_positional" for p in params)
        return _flag(plan, f"the call passes {hit.positional} positional "
                           f"argument(s) and the old signature has "
                           f"{len(named)} named one(s)"
                           + (": the rest went to *args, which has no name"
                              if has_star else ": the call was already "
                              "wrong"), flags)
    passed = named[:hit.positional]
    starts = [i for i, p in enumerate(passed) if p.name in moved]
    if not starts:
        by_name = sorted(moved & set(hit.keywords))
        if by_name:
            plan.reason = ("already passed by name: "
                           + ", ".join(by_name))
        else:
            plan.reason = ("the moved parameter(s) are not passed here: "
                           + _listed(moved))
        if flags:
            plan.reason += "; flagged: " + "; ".join(flags)
        return plan
    start = starts[0]
    run = passed[start:]
    for p in run:
        if p.kind != "positional_or_keyword":
            return _flag(plan, f"{p.name} was positional-only in the old "
                               "version: it has no name to pass", flags)
        if p.name in hit.keywords:
            return _flag(plan, f"{p.name} is passed by position and by "
                               "keyword: the call is already wrong", flags)
    plan.names = tuple(p.name for p in run)
    plan.start = start
    if flags:
        return _flag(plan, "", flags)
    plan.action = REWRITE
    plan.reason = "pass by name: " + ", ".join(plan.names)
    return plan


def _listed(names: set[str], cut: int = 6) -> str:
    """A few names, and how many more (Redis.__init__ moved 43)."""
    ordered = sorted(names)
    if len(ordered) <= cut:
        return ", ".join(ordered)
    return ", ".join(ordered[:cut]) + f" and {len(ordered) - cut} more"


def _flag(plan: Plan, reason: str, flags: list[str]) -> Plan:
    plan.action = FLAG
    parts = [reason] if reason else []
    if flags:
        parts.append("the function also carries a flag: " + "; ".join(flags))
    plan.reason = "; ".join(parts)
    return plan


PARAMETER_KINDS = (MOVED_KINDS | BLOCKING_KINDS
                   | {"PARAMETER_CHANGED_KIND", "PARAMETER_CHANGED_DEFAULT",
                      "PARAMETER_CHANGED_REQUIRED"})


def r1_plans(located: Located, changes: list[Change],
             api: OldApi | None, why_unread: str = "") -> list[Plan]:
    """One plan per resolved call that meets a function with R1 rows,
    decided from the parameter rows of EVERY function the call meets, so
    a flag under one defining path withholds the rewrite under another."""
    by_symbol: dict[str, list[Change]] = {}
    for c in changes:
        by_symbol.setdefault(c.symbol, []).append(c)
    by_id = {c.id: c for c in changes}
    plans: list[Plan] = []
    loaded = api if api is not None and api.loaded else None
    for h in located.hits:
        symbols = sorted({by_id[i].symbol for i in h.change_ids
                          if i in by_id})
        functions = [sym for sym in symbols
                     if any(c.kind in PARAMETER_KINDS for c in by_symbol[sym])]
        rows = [c for sym in functions for c in by_symbol[sym]]
        if not any(c.kind in MOVED_KINDS or c.kind == "PARAMETER_CHANGED_KIND"
                   for c in rows):
            continue
        # the signature: the first function the old version can show
        symbol, params, labels = functions[0], None, set()
        if loaded is not None:
            for sym in functions:
                params = loaded.parameters(sym)
                if params is not None:
                    symbol, labels = sym, loaded.labels(sym)
                    break
        why = why_unread if loaded is None else (
            "" if params is not None else
            "griffe sees no signature for it (a generated __init__, an "
            "alias it cannot resolve, or no such function in the old "
            "version)")
        plan = plan_r1(h, symbol, rows, params, labels, why)
        if plan is not None:
            plan.symbols = tuple(functions)
            plans.append(plan)
    return plans


# ------------------------------------------------------------- the edit

@dataclasses.dataclass
class FileDiff:
    path: str
    old: str
    new: str

    @property
    def text(self) -> str:
        """A unified diff `patch -p1` and `git apply` accept: every line
        terminated, and the `\\ No newline at end of file` marker where a
        file's last line has none (difflib leaves that line unterminated,
        which glues it to whatever follows)."""
        out = []
        for line in difflib.unified_diff(
                self.old.splitlines(keepends=True),
                self.new.splitlines(keepends=True),
                fromfile=f"a/{self.path}", tofile=f"b/{self.path}"):
            if line.endswith("\n"):
                out.append(line)
            else:
                out.append(line + "\n\\ No newline at end of file\n")
        return "".join(out)


def _rename(call: cst.Call, start: int, names: tuple[str, ...]) -> cst.Call:
    """The call with its positional arguments from `start` named."""
    equal = cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""))
    out, i = [], 0
    for a in call.args:
        if a.star or a.keyword is not None:
            out.append(a)
            continue
        if i >= start and i - start < len(names):
            a = a.with_changes(keyword=cst.Name(names[i - start]),
                               equal=equal)
        i += 1
        out.append(a)
    return call.with_changes(args=out)


class _Renamer(cst.CSTTransformer):
    """Applies the plans of one file; records each call's new text."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plans: dict[tuple, Plan], apply: bool):
        self.plans = plans
        self.apply = apply
        self.seen: dict[tuple, str] = {}
        self.module: cst.Module | None = None

    def visit_Module(self, node: cst.Module) -> None:
        self.module = node

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.Call:
        span = self.get_metadata(PositionProvider, original)
        key = (span.start.line, span.start.column + 1, span.end.line,
               span.end.column)
        plan = self.plans.get(key)
        if plan is None or not plan.names:
            return updated
        new = _rename(updated, plan.start, plan.names)
        self.seen[key] = self.module.code_for_node(new) if self.module \
            else ""
        return new if self.apply and plan.action == REWRITE else updated


def render(root: str | pathlib.Path, plans: list[Plan]
           ) -> list[FileDiff]:
    """Fill each plan's `would_be` from the project's source and return
    the diffs of the files R1 rewrites. Files are read, never written
    here; `apply` writes."""
    root = pathlib.Path(root).resolve()
    by_file: dict[str, dict[tuple, Plan]] = {}
    for p in plans:
        if p.names:
            h = p.hit
            key = (h.line, h.column, h.end_line, h.end_column)
            by_file.setdefault(h.file, {})[key] = p
    diffs: list[FileDiff] = []
    for rel, keyed in sorted(by_file.items()):
        text = (root / rel).read_bytes().decode("utf-8")
        bom = text.startswith("\ufeff")       # libcst drops it; kept here
        wrapper = MetadataWrapper(cst.parse_module(text[1:] if bom
                                                   else text),
                                  unsafe_skip_copy=True)
        renamer = _Renamer(keyed, apply=True)
        new_module = wrapper.visit(renamer)
        for key, plan in keyed.items():
            plan.would_be = renamer.seen.get(key)
        if any(p.action == REWRITE for p in keyed.values()):
            new_text = ("\ufeff" if bom else "") + new_module.code
            if new_text != text:
                diffs.append(FileDiff(rel, text, new_text))
    return diffs


def apply(root: str | pathlib.Path, diffs: list[FileDiff]) -> list[str]:
    """Write the rewritten files; the paths written."""
    root = pathlib.Path(root).resolve()
    written = []
    for d in diffs:
        (root / d.path).write_bytes(d.new.encode("utf-8"))
        written.append(d.path)
    return written
