"""
The rewrites (NOTES §38.4): R1, pass by name, and R2, pin the old default.
Nothing else is rewritten.

A rewrite is decided per CALL SITE from every row of the called function
in the range, never per row, and both rules decide in one pass, so
`Redis(...)` with 43 moved parameters and six changed defaults is one
decision and one edit. Each rule's part of the decision is kept (a
`Rule`: rewrite, withheld with the reason, or needs nothing), the report
prints the call as the diff will have it and then one line per rule, and
a rule that is withheld still says what it would be.

R1 takes a resolved call (locate's `call` or `method` hit) and the
function's rows; where the call passes by position an argument whose
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
is not.

R2 (NOTES §38.15) takes the same call and the function's
PARAMETER_CHANGED_DEFAULT rows (old and new value from the bundle) and
PARAMETER_CHANGED_REQUIRED rows (the old default read from the old
signature, since griffe's message carries none): a parameter the call
does not pass gets `name=<old default>` appended, when the old default is
a plain literal, `plain`: it parses with ast.literal_eval, it is not a
mutable container, also inside a tuple (a shared default and a fresh one
are different programs), and where the new default is a literal too the
two differ in value or type (`cosmetic`; griffe renders a constant
default with repr, so a row that differs only in spelling does not occur
in the data, 0 of 601 plain old values in the bundle, and the check is a
guard against a reader that compares text). The
pins go after the last argument in the old signature's order, as
`name=value`; a call that ends with a trailing comma on its own line gets
one pin per line with the arguments' indentation and the comma last; a
bare generator argument (`f(x for x in y)`) takes its parentheses, since
it is legal alone and not beside another argument.
What is not pinned is said with the old and new value: an old default
that is not a plain literal (redis's `retry=Retry(...)`), a mutable
container, a positional-only parameter (no name to pass).

R2 is withheld, with the reason, when:
  - the old signature could not be read, for any pin that is not already
    passed by name: whether a positional argument fills the parameter,
    and whether it takes a keyword at all, is unknown, and an edit is
    never made on a guess (R1 decides "no positional argument" without
    the signature because that outcome is nothing);
  - the call uses `*` or `**` unpacking: it may already pass the name;
  - the call is already wrong: more positional arguments than the old
    signature has named ones with no `*args`, or any parameter passed
    by position and by keyword;
  - the function also carries a flag, as for R1, including two R2 finds
    that withhold R1 as well: a parameter now required whose old default
    cannot be pinned (the call will not run on the new version whatever
    else is done), and one name resolving to two functions of the old
    version with different signatures; the pins are printed as what
    they would be.
R2 is "left to you", a third outcome, when there is nothing to pin and a
row it cannot pin: the row is said, with its values, so the user can
set it (redis's `retry`).

Both rules: libcst makes the edit and keeps the file's formatting,
comments, quotes, line endings and byte-order mark, so the diff shows the
lines the rules change and nothing else; a second run after `--apply`
finds every call it rewrote passing by name and pinned, and rewrites
nothing.

One premise the data cannot check (the review of 22 Oct): R1 assumes
that no parameter BEFORE the first moved one vanished without a row.
griffe reports no row for a parameter removed when the new signature
gains `*args` and `**kwargs` together (it counts as swallowed), and the
parameters after it as moved; `swallow(a, b, c)` -> `swallow(a, c, *args,
**kwargs)` is one MOVED row on `c`, and `swallow(1, 2, 3)` ->
`swallow(1, 2, c=3)` raises on the new version. The bundle's one-line
message carries no positions, so the tool cannot see it; §38.6's hand
check judges such a rewrite, and §38.11's ingest change (griffe's
`details`, with the positions) would let R1 withhold it. R2's own
premise is §38.4's: the call now states the value it relied on; whether
the new release treats that value as the old one did is the library's
business, and every pin is printed with "delete this argument to accept
the new default" beside it.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import pathlib
import re

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from fixer.bundle import Change
from fixer.locate import RESOLVED, Hit, Located
from fixer.oldapi import OldApi, Param

REWRITE, FLAG, LEFT, NOTHING, NONE = "rewrite", "flag", "left", "nothing", ""

# The rows R1 acts on, and the rows that withhold both rules (§38.4).
MOVED_KINDS = {"PARAMETER_MOVED"}
BLOCKING_KINDS = {"PARAMETER_REMOVED", "PARAMETER_ADDED_REQUIRED"}
R2_KINDS = {"PARAMETER_CHANGED_DEFAULT", "PARAMETER_CHANGED_REQUIRED"}
KIND_CHANGE = re.compile(r"Parameter kind was changed: (.+?) -> (.+?)\s*$")
NAMED = ("positional_only", "positional_or_keyword")
# A changed value longer than this is cut in a reason; the diff carries
# the whole value (fixer/report.py cuts the same way).
VALUE_CUT = 72


def kind_change(message: str) -> tuple[str, str] | None:
    """griffe's old and new parameter kinds out of a PARAMETER_CHANGED_KIND
    message (`... -> keyword-only`), or None."""
    m = KIND_CHANGE.search(message)
    return (m.group(1), m.group(2)) if m else None


# ------------------------------------------------------------ R2's values

def _value(text: str):
    """The Python value of a printed default, or raise."""
    return ast.literal_eval(text.strip())


def why_not_plain(text: str) -> str | None:
    """Why an old default cannot be pinned, in words, or None when it is a
    plain literal (§38.4, §38.15): a value ast.literal_eval does not read
    (a name, a call, an expression, `set()`) is not a plain value; a list,
    dict or set literal, empty or not, also inside a tuple, is a mutable
    container."""
    try:
        value = _value(text)
    except Exception:  # noqa: BLE001 - not a literal, whatever the reason
        return "is not a plain value"
    if _mutable(value):
        return "is a mutable container"
    return None


def _mutable(value) -> bool:
    """A list, dict or set, at any depth inside a tuple."""
    if isinstance(value, (list, dict, set)):
        return True
    return isinstance(value, tuple) and any(_mutable(v) for v in value)


def plain(text: str) -> bool:
    return why_not_plain(text) is None


def cosmetic(old: str, new: str) -> bool:
    """True when old and new are literals of the same value and type,
    spelled differently (`'a'` and `"a"`, `1e3` and `1000.0`): griffe
    compares source text, so these are rows, and nothing is pinned."""
    try:
        a, b = _value(old), _value(new)
    except Exception:  # noqa: BLE001
        return False
    return type(a) is type(b) and a == b


def _short(value: str) -> str:
    if len(value) <= VALUE_CUT:
        return value
    return value[:VALUE_CUT - 1] + "…"


# ------------------------------------------------------------ the plan

@dataclasses.dataclass
class Rule:
    """One rule's part of a call's decision: rewrite, flag (withheld, the
    reason says why) or nothing (and why); NONE when the rule has no row
    on the functions the call meets. `would_be` is the call with this
    rule's edit, for a withheld rule that could compute one."""
    action: str = NONE
    reason: str = ""
    would_be: str | None = None


@dataclasses.dataclass
class Plan:
    """The decision for one call site, from every parameter row of every
    function the call meets (one, unless an export path names two
    defining paths inside the range), by both rules."""
    hit: Hit
    # the function whose signature was read, by its defining path
    symbol: str
    # every function with parameter rows the call meets; the decision
    # prints under each of their entries
    symbols: tuple[str, ...] = ()
    rows: tuple[int, ...] = ()       # the change ids behind the decision
    names: tuple[str, ...] = ()      # R1's edit: the parameters the run names
    start: int = 0                   # first positional argument renamed
    pins: tuple[str, ...] = ()       # R2's edit: `name=value`, signature order
    r1: Rule = dataclasses.field(default_factory=Rule)
    r2: Rule = dataclasses.field(default_factory=Rule)
    # the call's text after the rules that rewrite, from the source
    would_be: str | None = None

    @property
    def action(self) -> str:
        """The call's headline: rewritten if either rule rewrites, else
        withheld if either is, else left to the user if R2 found only
        rows it cannot pin, else nothing."""
        actions = (self.r1.action, self.r2.action)
        for a in (REWRITE, FLAG, LEFT):
            if a in actions:
                return a
        return NOTHING

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


def function_flags(rows: list[Change]) -> tuple[set[str], list[str]]:
    """R1's moved parameters, and the flags that withhold both rules: a
    parameter removed, one added as required, one made positional-only,
    or a change of kind neither rule covers."""
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
    flags.sort()
    return moved, flags


def _flag(rule: Rule, reason: str, flags: list[str]) -> Rule:
    rule.action = FLAG
    parts = [reason] if reason else []
    if flags:
        parts.append("the function also carries a flag: " + "; ".join(flags))
    rule.reason = "; ".join(parts)
    return rule


def _nothing(rule: Rule, reason: str, flags: list[str]) -> Rule:
    rule.action = NOTHING
    rule.reason = reason + ("; flagged: " + "; ".join(flags) if flags else "")
    return rule


def _listed(names, cut: int = 6) -> str:
    """A few names, and how many more (Redis.__init__ moved 43)."""
    ordered = sorted(names)
    if len(ordered) <= cut:
        return ", ".join(ordered)
    return ", ".join(ordered[:cut]) + f" and {len(ordered) - cut} more"


def _named(params: list[Param], hit: Hit, labels: set[str]) -> list[Param]:
    """The old signature's named parameters the call's positional
    arguments fill, in order, after the self/cls offset."""
    return [p for p in params if p.kind in NAMED][offset(hit, labels):]


def plan_r1(plan: Plan, hit: Hit, rows: list[Change], params,
            labels: set[str], why_unread: str = "",
            extra_flags: tuple[str, ...] = ()) -> Rule:
    """R1 for one resolved call whose rows in the range are `rows` and
    whose old parameters are `params` (None when unreadable). Fills
    `plan.names` and `plan.start` where there is an edit, withheld or
    not. `extra_flags` are flags found outside the rows (a now-required
    parameter R2 cannot pin; two old signatures for one name), which
    withhold like the rows'. A Rule with action NONE when R1 has no row
    on the function."""
    rule = Rule()
    moved, flags = function_flags(rows)
    flags = sorted(flags + list(extra_flags))
    if not moved:
        return rule
    if hit.star_args:
        return _flag(rule, "the call unpacks `*` or `**`: the positions "
                           "are unknown", flags)
    # Two decisions need no signature: every moved parameter is already
    # passed by name, or nothing is passed by position at all.
    if moved <= set(hit.keywords):
        return _nothing(rule, "already passed by name: "
                        + ", ".join(sorted(moved)), flags)
    if hit.positional == 0:
        return _nothing(rule, "the moved parameter(s) are not passed here: "
                        + _listed(moved), flags)
    if params is None:
        return _flag(rule, "the old signature could not be read"
                           + (f": {why_unread}" if why_unread else ""),
                     flags)
    named = _named(params, hit, labels)
    if hit.positional > len(named):
        has_star = any(p.kind == "var_positional" for p in params)
        return _flag(rule, f"the call passes {hit.positional} positional "
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
            return _nothing(rule, "already passed by name: "
                            + ", ".join(by_name), flags)
        return _nothing(rule, "the moved parameter(s) are not passed here: "
                        + _listed(moved), flags)
    start = starts[0]
    run = passed[start:]
    for p in run:
        if p.kind != "positional_or_keyword":
            return _flag(rule, f"{p.name} was positional-only in the old "
                               "version: it has no name to pass", flags)
        if p.name in hit.keywords:
            return _flag(rule, f"{p.name} is passed by position and by "
                               "keyword: the call is already wrong", flags)
    plan.names = tuple(p.name for p in run)
    plan.start = start
    if flags:
        return _flag(rule, "", flags)
    rule.action = REWRITE
    rule.reason = ", ".join(plan.names)
    return rule


def plan_r2(plan: Plan, hit: Hit, rows: list[Change], params,
            labels: set[str], why_unread: str = "",
            extra_flags: tuple[str, ...] = ()) -> tuple[Rule, list[str]]:
    """R2 for one resolved call: the pins, in the old signature's order,
    into `plan.pins` where there is an edit, withheld or not; the Rule
    says what was pinned, what was not and why, and what is already
    passed; and the flags R2 itself found, which R1 must honour too: a
    parameter now required whose old default cannot be pinned, so the
    call will not run on the new version whatever else is done. A Rule
    with action NONE when R2 has no row on the function."""
    rule = Rule()
    r2rows = [c for c in rows if c.kind in R2_KINDS]
    if not r2rows:
        return rule, []
    _moved, flags = function_flags(rows)
    flags = sorted(flags + list(extra_flags))
    found: list[str] = []
    by_name = {p.name: p for p in params} if params is not None else {}
    order = {p.name: i for i, p in enumerate(params or [])}
    # the parameters the call's positional arguments fill, by the old
    # signature; and whether it passes more than the signature can take
    by_position: set[str] = set()
    overflow = ""
    if params is not None:
        named = _named(params, hit, labels)
        if hit.positional > len(named) and not any(
                p.kind == "var_positional" for p in params):
            overflow = (f"the call passes {hit.positional} positional "
                        f"argument(s) and the old signature has "
                        f"{len(named)} named one(s): the call was already "
                        "wrong")
        by_position = {p.name for p in named[:hit.positional]}

    # Each row: a candidate (name, old value, "now ..."), already passed,
    # a row that cannot be pinned (name, why), or a cosmetic row. Rows in
    # the old signature's order where it is known, else as they come.
    candidates: list[tuple[str, str, str]] = []
    passed: list[str] = []
    not_pinned: list[tuple[str, str]] = []
    spelling: list[str] = []
    unread: list[str] = []             # now-required rows, signature unread
    for c in sorted(r2rows, key=lambda c: order.get(c.sub_target, 10**6)):
        name = c.sub_target
        if name in hit.keywords or name in by_position:
            passed.append(name)
            continue
        required = c.kind == "PARAMETER_CHANGED_REQUIRED"
        if required:
            if params is None:
                unread.append(name)
                continue
            p = by_name.get(name)
            old = p.default if p is not None else None
            if old is None:
                found.append(f"{name} is now required in {c.version_to} and "
                             "cannot be pinned (it had no default in the old "
                             "version the tool read)")
                continue
            now, new_text = "now required", "required"
        else:
            old, new_text = c.old_value, c.new_value
            if old is None or new_text is None:
                not_pinned.append((name, "old and new value could not be "
                                         "read from griffe's message"))
                continue
            if cosmetic(old, new_text):
                spelling.append(f"{name} ({_short(old)} -> "
                                f"{_short(new_text)})")
                continue
            now = f"now {_short(new_text)}"
        why = why_not_plain(old)
        p = by_name.get(name)
        if why is None and params is not None and p is None:
            why = "is not a parameter of the old signature the tool read"
        if why is None and p is not None and p.kind == "positional_only":
            why = "is positional-only in the old version: no name to pass"
        if why is not None:
            if required:
                # the call will not run on the new version whatever else
                # is done: a flag, for both rules
                found.append(f"{name} is now required in {c.version_to} and "
                             f"cannot be pinned (old default {_short(old)} "
                             f"{why})")
            else:
                not_pinned.append((name, f"old default {_short(old)} {why}; "
                                         f"{now}"))
            continue
        candidates.append((name, old, now))

    flags = sorted(flags + found)

    def tail(parts: list[str]) -> str:
        return "; ".join(part for part in parts if part)

    not_pinned_text = (", ".join(f"{n} ({why})" for n, why in not_pinned)
                       if not_pinned else "")
    spelling_text = ("changed only in spelling: " + ", ".join(spelling)
                     if spelling else "")
    passed_text = ("already passed: " + ", ".join(sorted(passed))
                   if passed else "")

    if hit.star_args:
        return _flag(rule, "the call unpacks `*` or `**`: it may already "
                           "pass " + (_listed(n for n, _, _ in candidates)
                                      if candidates else "the parameter(s)"),
                     flags), found
    if overflow:
        return _flag(rule, overflow, flags), found
    if params is None and (candidates or unread):
        return _flag(rule, "the old signature could not be read"
                           + (f": {why_unread}" if why_unread else ""),
                     flags), found
    if candidates:
        wrong = sorted(by_position & set(hit.keywords))
        if wrong:
            return _flag(rule, f"{wrong[0]} is passed by position and by "
                               "keyword: the call is already wrong",
                         flags), found
    if not candidates:
        if flags:
            return _flag(rule, "", flags), found
        if not_pinned:
            # rows only the user can set: said, with their values
            rule.action = LEFT
            rule.reason = tail([not_pinned_text, passed_text, spelling_text])
            return rule, found
        return _nothing(rule, tail([passed_text, spelling_text])
                        or "nothing to pin", flags), found

    plan.pins = tuple(f"{n}={old}" for n, old, _ in candidates)
    pinned_text = ", ".join(f"{n}={_short(old)} ({now})"
                            for n, old, now in candidates)
    if flags:
        return _flag(rule, "", flags), found
    rule.action = REWRITE
    rule.reason = tail([pinned_text,
                        "not pinned: " + not_pinned_text if not_pinned else "",
                        passed_text, spelling_text])
    return rule, found


PARAMETER_KINDS = (MOVED_KINDS | BLOCKING_KINDS | R2_KINDS
                   | {"PARAMETER_CHANGED_KIND"})


def decide(located: Located, changes: list[Change],
           api: OldApi | None, why_unread: str = "") -> list[Plan]:
    """One plan per resolved call that meets a function with a row either
    rule acts on, decided from the parameter rows of EVERY function the
    call meets, so a flag under one defining path withholds the rewrite
    under another."""
    by_symbol: dict[str, list[Change]] = {}
    for c in changes:
        by_symbol.setdefault(c.symbol, []).append(c)
    by_id = {c.id: c for c in changes}
    plans: list[Plan] = []
    loaded = api if api is not None and api.loaded else None
    for h in located.hits:
        if h.confidence != RESOLVED or h.use not in ("call", "method"):
            continue
        symbols = sorted({by_id[i].symbol for i in h.change_ids
                          if i in by_id})
        functions = [sym for sym in symbols
                     if any(c.kind in PARAMETER_KINDS for c in by_symbol[sym])]
        rows = [c for sym in functions for c in by_symbol[sym]]
        if not any(c.kind in MOVED_KINDS or c.kind == "PARAMETER_CHANGED_KIND"
                   or c.kind in R2_KINDS for c in rows):
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
        # one name resolving to two functions the old version has, with
        # different signatures (a compatibility shim): the edit computed
        # on one may be wrong for the other, so both rules are withheld
        extra: list[str] = []
        if params is not None:
            shape = [(p.name, p.kind) for p in params]
            for sym in functions:
                if sym == symbol:
                    continue
                other = loaded.parameters(sym)
                if other is not None and [(p.name, p.kind)
                                          for p in other] != shape:
                    extra.append(f"the call resolves to {symbol} and {sym}, "
                                 "whose old signatures differ")
        plan = Plan(h, symbol, tuple(functions),
                    tuple(sorted(c.id for c in rows)))
        plan.r2, found = plan_r2(plan, h, rows, params, labels, why,
                                 tuple(extra))
        plan.r1 = plan_r1(plan, h, rows, params, labels, why,
                          tuple(extra + found))
        if plan.r1.action == NONE and plan.r2.action == NONE:
            continue
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
                _lines(self.old), _lines(self.new),
                fromfile=f"a/{self.path}", tofile=f"b/{self.path}"):
            if line.endswith("\n"):
                out.append(line)
            else:
                out.append(line + "\n\\ No newline at end of file\n")
        return "".join(out)


def _lines(text: str) -> list[str]:
    """The file's lines, ending in `\n` where they do, split on `\n`
    alone: str.splitlines also breaks at a form feed, a vertical tab and
    a few other characters Python keeps inside a line, which would put
    the no-newline marker in the middle of a hunk."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


EQUAL = cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""),
                        whitespace_after=cst.SimpleWhitespace(""))


def _rename(call: cst.Call, start: int, names: tuple[str, ...]) -> cst.Call:
    """The call with its positional arguments from `start` named."""
    out, i = [], 0
    for a in call.args:
        if a.star or a.keyword is not None:
            out.append(a)
            continue
        if i >= start and i - start < len(names):
            a = a.with_changes(keyword=cst.Name(names[i - start]),
                               equal=EQUAL)
        i += 1
        out.append(a)
    return call.with_changes(args=out)


def _pin(call: cst.Call, pins: tuple[str, ...]) -> cst.Call:
    """The call with `name=value` arguments appended, in the order given.
    A call whose last argument ends with a comma on its own line (the
    trailing-comma style) gets one pin per line with the arguments'
    indentation and keeps the trailing comma last; any other call gets
    the pins joined to its last line."""
    args = list(call.args)
    new = []
    for text in pins:
        name, value = text.split("=", 1)
        new.append(cst.Arg(value=cst.parse_expression(value),
                           keyword=cst.Name(name), equal=EQUAL))
    if not args:
        return call.with_changes(args=new)
    last = args[-1]
    if not isinstance(last.comma, cst.Comma):
        # the pins join the last argument's line; the whitespace that
        # ran from it to `)` (a line break, in a call spanning lines)
        # moves after the last pin, so `)` stays where it was
        after = last.whitespace_after_arg
        args[-1] = last.with_changes(
            whitespace_after_arg=cst.SimpleWhitespace(""))
        new[-1] = new[-1].with_changes(whitespace_after_arg=after)
        return call.with_changes(args=args + new)
    # the trailing-comma style: the whitespace that precedes an argument
    # is the previous comma's, or the call's own before its first
    # argument; a comment riding on it (`1,  # the port`) is not copied
    # to each new line, and a same-line comma takes one space
    if len(args) >= 2 and isinstance(args[-2].comma, cst.Comma):
        before = args[-2].comma.whitespace_after
    else:
        before = call.whitespace_before_args
    before = _uncommented(before)
    if isinstance(before, cst.SimpleWhitespace):
        before = cst.SimpleWhitespace(" ")
    lead = cst.Comma(whitespace_after=before)
    closing = last.comma
    own = last.comma.whitespace_after
    if (isinstance(own, cst.ParenthesizedWhitespace)
            and own.first_line.comment is not None
            and isinstance(before, cst.ParenthesizedWhitespace)):
        # `retries=1,  # retry once` keeps its comment: the comma stays
        # on its argument, with the line break's indentation made an
        # argument's, and the closing comma carries no comment
        args[-1] = last.with_changes(comma=last.comma.with_changes(
            whitespace_after=own.with_changes(indent=before.indent,
                                              last_line=before.last_line)))
        closing = cst.Comma(whitespace_after=_uncommented(own))
    else:
        args[-1] = last.with_changes(comma=lead)
    for k, a in enumerate(new):
        new[k] = a.with_changes(comma=closing if k == len(new) - 1
                                else lead)
    return call.with_changes(args=args + new)


def _uncommented(ws):
    """The same line break and indentation, without a comment or blank
    lines on it."""
    if isinstance(ws, cst.ParenthesizedWhitespace):
        return ws.with_changes(
            first_line=cst.TrailingWhitespace(
                whitespace=cst.SimpleWhitespace(""), comment=None,
                newline=ws.first_line.newline),
            empty_lines=[])
    return ws


def _parenthesised(call: cst.Call) -> cst.Call:
    """A bare generator argument, `f(x for x in y)`, is legal only alone:
    it takes its parentheses once it is named or another argument
    follows."""
    args = list(call.args)
    for k, a in enumerate(args):
        if isinstance(a.value, cst.GeneratorExp) and not a.value.lpar:
            args[k] = a.with_changes(value=a.value.with_changes(
                lpar=[cst.LeftParen()], rpar=[cst.RightParen()]))
    return call.with_changes(args=args)


def _edit(call: cst.Call, plan: Plan, r1: bool, r2: bool) -> cst.Call:
    if (r1 and plan.names) or (r2 and plan.pins):
        call = _parenthesised(call)
    if r1 and plan.names:
        call = _rename(call, plan.start, plan.names)
    if r2 and plan.pins:
        call = _pin(call, plan.pins)
    return call


class _Rewriter(cst.CSTTransformer):
    """Applies the plans of one file; records each call's new text, and
    for a withheld rule the text with that rule's edit alone."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plans: dict[tuple, Plan]):
        self.plans = plans
        self.seen: dict[tuple, dict[str, str | None]] = {}
        self.module: cst.Module | None = None

    def visit_Module(self, node: cst.Module) -> None:
        self.module = node

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.Call:
        span = self.get_metadata(PositionProvider, original)
        key = (span.start.line, span.start.column + 1, span.end.line,
               span.end.column)
        plan = self.plans.get(key)
        if plan is None or not (plan.names or plan.pins):
            return updated
        code = (self.module.code_for_node if self.module
                else (lambda n: ""))
        do1, do2 = plan.r1.action == REWRITE, plan.r2.action == REWRITE
        actual = _edit(updated, plan, do1, do2)
        # a withheld rule keeps its edit only under the function's flags,
        # which withhold both rules, so each would-be is that rule's edit
        # alone
        self.seen[key] = {
            "actual": code(actual),
            "r1": (code(_edit(updated, plan, True, False))
                   if plan.r1.action == FLAG and plan.names else None),
            "r2": (code(_edit(updated, plan, False, True))
                   if plan.r2.action == FLAG and plan.pins else None),
        }
        return actual


def render(root: str | pathlib.Path, plans: list[Plan]
           ) -> list[FileDiff]:
    """Fill each plan's `would_be` (and each withheld rule's) from the
    project's source and return the diffs of the files the rules rewrite.
    Files are read, never written here; `apply` writes."""
    root = pathlib.Path(root).resolve()
    by_file: dict[str, dict[tuple, Plan]] = {}
    for p in plans:
        if p.names or p.pins:
            h = p.hit
            key = (h.line, h.column, h.end_line, h.end_column)
            by_file.setdefault(h.file, {})[key] = p
    diffs: list[FileDiff] = []
    for rel, keyed in sorted(by_file.items()):
        text = (root / rel).read_bytes().decode("utf-8")
        bom = text.startswith("﻿")       # libcst drops it; kept here
        wrapper = MetadataWrapper(cst.parse_module(text[1:] if bom
                                                   else text),
                                  unsafe_skip_copy=True)
        rewriter = _Rewriter(keyed)
        new_module = wrapper.visit(rewriter)
        for key, plan in keyed.items():
            seen = rewriter.seen.get(key)
            if seen is None:
                continue
            plan.r1.would_be = seen["r1"]
            plan.r2.would_be = seen["r2"]
            if plan.action == REWRITE:
                plan.would_be = seen["actual"]
        if any(p.action == REWRITE for p in keyed.values()):
            new_text = ("﻿" if bom else "") + new_module.code
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
