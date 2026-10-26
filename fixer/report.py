"""
The report (NOTES §38.5): rank first, then what the project touches.

Two parts. The first is every change the locate step found in the
project, in the site's order (path length, the ranker that ships, then
name), one entry per changed FUNCTION or object with every row of it
under it, since a call to `Redis(...)` with 49 moved parameters is one
call site and not 49; under the rows, the project's lines, each with its
confidence (resolved, referenced, unverified; §38.3) and how the name is
used, and under a call the rules decided on (§38.4, §38.15), the
decision in three lines at most: rewritten and the call as the diff will
have it, withheld, or needs nothing; then one line per rule that has a
row on the function, with what it did (R1 the names, R2 the pins with
their new values and what it did not pin and why), or why it is withheld
and what it would be, or why nothing is needed. The second part is the
rest of the upgrade's changes, ranked, one line each, cut at SHOW_REST
lines with the count of the rest, since sglang 0.5.10 alone has 2,011.
Private names are reported when the project uses them, marked private.
The footer counts each rule's decisions and credits griffe.

`rank_report` is the first week's report, kept for `--rank-only`: the
releases crossed and the ranked list, the project unread.
"""

from __future__ import annotations

import libcst as cst

from fixer.bundle import Change, Release
from fixer.locate import REFERENCED, RESOLVED, UNVERIFIED, Hit, Located
from fixer.rewrite import FLAG, LEFT, NONE, NOTHING, REWRITE, Plan, Rule

SHOW = 60        # the ranked list of --rank-only
SHOW_REST = 200  # the untouched changes under part two
# A changed value longer than this is cut in the one-line forms (redis's
# REDIS_ALLOWED_KEYS is a 44-name tuple); the bundle holds the whole text,
# and R2 pins the whole text in the diff (fixer/rewrite.py cuts its
# reasons the same way).
VALUE_CUT = 72


def describe_change(c: Change) -> str:
    """One line per change for a ranked list."""
    what = c.symbol + (f" ({c.sub_target})" if c.sub_target else "")
    return f"{what}   {c.kind}{_values(c)}   used by {c.alias_user_count}" \
           f"{'   private' if c.is_private else ''}"


def _short(value: str) -> str:
    if len(value) <= VALUE_CUT:
        return value
    return value[:VALUE_CUT - 1] + "\u2026"


def _values(c: Change) -> str:
    if c.old_value is not None and c.new_value is not None:
        return f"   {_short(c.old_value)} -> {_short(c.new_value)}"
    return ""


def describe_row(c: Change) -> str:
    """One row of a function, without the function's name."""
    sub = f" ({c.sub_target})" if c.sub_target else ""
    return f"{c.kind}{sub}{_values(c)}"


def describe_hit(h: Hit) -> str:
    where = f"{h.file}:{h.line}:{h.column}"
    return f"{where:<28} {h.confidence:<11} {h.use:<9} {h.source}"


def header(package: str, frm: str, to: str, rels: list[Release],
           n_changes: int, built_at: str) -> list[str]:
    lines = [f"{package} {frm} -> {to}: {len(rels)} release(s) crossed, "
             f"{n_changes} change(s); bundle built {built_at}"]
    for r in rels:
        lines.append(f"  {r.version:<14} {r.describe()}")
    not_compared = [r for r in rels if not r.compared]
    if not_compared:
        lines.append("  ** " + str(len(not_compared)) + " release(s) in the "
                     "range were not compared, so this list cannot be "
                     "complete for them.")
    return lines


def rank_report(package: str, frm: str, to: str, rels: list[Release],
                changes: list[Change], built_at: str, note: str = "") -> str:
    """The ranked list alone (--rank-only): the project is not read."""
    lines = header(package, frm, to, rels, len(changes), built_at)
    if note:
        lines.append(note)
    lines.append("")
    if not changes:
        lines.append("no changes recorded in these releases.")
    else:
        lines.append("changes, in the site's order (shortest public path "
                     "first, then by name); \"used by\" counts the scanned "
                     "packages that\n  name the symbol by any of its paths "
                     "(NOTES §24):")
        for i, c in enumerate(changes[:SHOW], 1):
            lines.append(f"  {i:>3}  {describe_change(c)}")
        if len(changes) > SHOW:
            lines.append(f"       ... and {len(changes) - SHOW:,} more")
    lines.append("")
    lines.append("rank only: the project was not read (--rank-only).")
    lines.append("changes found with griffe.")
    return "\n".join(lines)


def groups(changes: list[Change]) -> list[tuple[str, list[Change]]]:
    """The changes by symbol, in the order the first row of each appears
    in the ranked list."""
    out: dict[str, list[Change]] = {}
    for c in changes:
        out.setdefault(c.symbol, []).append(c)
    return list(out.items())


class _NoComments(cst.CSTTransformer):
    """Comments out, and the line breaks inside a call's parentheses
    folded, so a call spanning lines prints as one."""

    def _flat(self, ws, after):
        if isinstance(ws, cst.ParenthesizedWhitespace):
            return cst.SimpleWhitespace(after)
        return ws

    def leave_Arg(self, original, updated):
        return updated.with_changes(
            whitespace_after_arg=self._flat(updated.whitespace_after_arg,
                                            ""))

    def leave_Comma(self, original, updated):
        return updated.with_changes(
            whitespace_after=self._flat(updated.whitespace_after, " "))

    def leave_Call(self, original, updated):
        args = list(updated.args)
        if args and isinstance(args[-1].comma, cst.Comma):
            args[-1] = args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        return updated.with_changes(
            args=args,
            whitespace_before_args=self._flat(updated.whitespace_before_args,
                                              ""))

    def leave_TrailingWhitespace(self, original, updated):
        return updated.with_changes(comment=None)

    def leave_EmptyLine(self, original, updated):
        return updated.with_changes(comment=None)


def one_line(text: str) -> str:
    """A call's text on one line: a call spanning lines folded into
    single spaces, its comments left out (folded in, `# the port db=2)`
    would read as one comment), a CRLF ending gone."""
    try:
        expr = cst.parse_expression(text)
        text = cst.Module([]).code_for_node(expr.visit(_NoComments()))
    except Exception:  # noqa: BLE001 - not an expression: fold as it is
        pass
    return " ".join(text.split())


RULES = (("R1", "pass by name"), ("R2", "pin the old default"))


def describe_rule(tag: str, name: str, rule: Rule) -> str | None:
    """One rule's line under a call, or None when the rule has no row on
    the function."""
    if rule.action == NONE:
        return None
    if rule.action == REWRITE:
        text = f"{tag}, {name}: {rule.reason}"
        if tag == "R2":
            text += "; delete an argument to accept its new default"
        return text
    if rule.action == FLAG:
        text = f"{tag} withheld: {rule.reason}"
        if rule.would_be:
            text += f" | would be: {one_line(rule.would_be)}"
        return text
    if rule.action == LEFT:
        return f"{tag} left to you: {rule.reason}"
    return f"{tag}: {rule.reason}"


def describe_plan(p: Plan) -> list[str]:
    """The decision under a call's line: the call as the diff will have
    it, withheld, or needs nothing; then each rule's line."""
    if p.action == REWRITE:
        lines = [f"-> rewritten: {one_line(p.would_be or '')}"]
    elif p.action == FLAG:
        lines = ["-> withheld"]
    elif p.action == LEFT:
        lines = ["-> left to you"]
    else:
        lines = ["-> needs nothing"]
    for (tag, name), rule in zip(RULES, (p.r1, p.r2)):
        text = describe_rule(tag, name, rule)
        if text is not None:
            lines.append(text)
    return lines


def summary(plans: list[Plan], diff_note: str) -> str:
    """The footer: each rule's counts over the calls it has a row on."""
    parts = []
    for (tag, name), attr, verb in zip(RULES, ("r1", "r2"),
                                       ("rewritten", "pinned")):
        n = {a: sum(1 for p in plans if getattr(p, attr).action == a)
             for a in (REWRITE, FLAG, LEFT, NOTHING)}
        parts.append(f"{tag}, {name}: {n[REWRITE]} call(s) {verb}, "
                     f"{n[FLAG]} withheld, "
                     + (f"{n[LEFT]} left to you, " if attr == "r2" else "")
                     + f"{n[NOTHING]} need nothing")
    text = ". ".join(parts)
    if any(p.action == REWRITE for p in plans):
        return text + f"; {diff_note}."
    return text + "."


def report(package: str, frm: str, to: str, rels: list[Release],
           changes: list[Change], located: Located, built_at: str,
           project: str, env_lines: list[str],
           plans: list[Plan] | None = None,
           diff_note: str = "the diff follows") -> str:
    plans = plans or []
    by_key = {p.key: p for p in plans}
    lines = header(package, frm, to, rels, len(changes), built_at)
    lines.append("")
    roots = ", ".join(located.roots) or package
    lines.append(f"project {project}: {located.files_read} file(s) read, "
                 f"{located.files_parsed} mention the import root(s) "
                 f"{roots}")
    if located.files_via_project:
        lines.append(f"  {located.files_via_project} file(s) import the "
                     "project's own modules and never the package: not "
                     "read for its names (a project module that re-exports "
                     "them is not followed).")
    if located.skipped_dirs:
        lines.append("  not entered (environments, build output): "
                     + ", ".join(located.skipped_dirs))
    lines += [f"  {s}" for s in env_lines]
    by_change = located.by_change()
    touched = [(sym, rows) for sym, rows in groups(changes)
               if any(c.id in by_change for c in rows)]
    n_touched = sum(1 for c in changes if c.id in by_change)
    counts = located.counts()
    lines.append(f"  {located.lines()} line(s) use {n_touched} of the "
                 f"{len(changes)} change(s), {len(located.hits)} hit(s): "
                 f"{counts[RESOLVED]} resolved, {counts[REFERENCED]} "
                 f"referenced, {counts[UNVERIFIED]} unverified.")
    lines.append("")

    if touched:
        lines.append("PART 1  changes your code uses, in the site's order "
                     "(shortest public path first, then by name):")
        for i, (sym, rows) in enumerate(touched, 1):
            used = max(c.alias_user_count for c in rows)
            private = "   private" if any(c.is_private for c in rows) else ""
            hits: dict[tuple, Hit] = {}
            for c in rows:
                for h in by_change.get(c.id, ()):
                    hits[(h.file, h.line, h.column, h.use, h.confidence)] = h
            n_lines = len({(h.file, h.line) for h in hits.values()})
            lines.append("")
            lines.append(f"  {i:>3}  {sym}   used by {used}{private}   "
                         f"({n_lines} line(s))")
            for c in rows:
                lines.append(f"         {describe_row(c)}")
            for key in sorted(hits):
                lines.append(f"       {describe_hit(hits[key])}")
                plan = by_key.get(key)
                if plan is not None and sym in plan.symbols:
                    first, *rest = describe_plan(plan)
                    lines.append(" " * 36 + first)
                    lines.extend(" " * 39 + text for text in rest)
    else:
        lines.append("PART 1  no line of the project names a changed "
                     "symbol (see the notes below for what the tool "
                     "cannot see).")
    lines.append("")

    rest = [c for c in changes if c.id not in by_change]
    lines.append(f"PART 2  the upgrade's other {len(rest)} change(s), "
                 "ranked, which your code does not name:")
    for c in rest[:SHOW_REST]:
        lines.append(f"       {describe_change(c)}")
    if len(rest) > SHOW_REST:
        lines.append(f"       ... and {len(rest) - SHOW_REST:,} more")
    lines.append("")

    if located.notes:
        lines.append("files the tool cannot see through, or did not read:")
        for n in located.notes:
            lines.append(f"  {n.file}: {n.text}")
    lines.append("getattr, strings and importlib are invisible to this tool. "
                 "In the files that import the package, a method called on "
                 "an object whose class the tool does not know is listed as "
                 "unverified; an object made by another package's call or "
                 "bound to a literal is not the package's and is not listed; "
                 "a project module that re-exports the package's names is "
                 "not followed.")
    lines.append(summary(plans, diff_note))
    lines.append("changes found with griffe.")
    return "\n".join(lines)
