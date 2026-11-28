"""
Does the fixer name the right library function for a call? A second,
separate check of the locate step on the calls that were read (NOTES
§39): the 230 of docs/eval/verdicts.csv.

    python scripts/eval_locate_check.py
    python scripts/eval_locate_check.py --root data/eval --record docs/eval/verdicts.csv
    python scripts/eval_locate_check.py --selftest

Reads  docs/eval/verdicts.csv      the record: for every read call its
                                   file, line and column, and the symbol
                                   the tool named for it
       data/eval/trees/            the sample packages' source, as
                                   scripts/eval_run.py fetch unpacked it
       data/eval/old/              the old version of each changed
                                   package (scripts/eval_run.py old)
Writes data/eval/locate_check.csv  one row per call: confirmed, or to be
                                   read by hand, how it was settled, and
                                   another definition of the same name in
                                   the library, where there is one

None of the fixer's code is used: nothing is imported from fixer/. For
each call one question is asked: is the thing called at that place the
library function the record names? It is answered in four steps.

  1. The call is found in the sample package's file with Python's ast
     module, by its line and column.
  2. A call written through an imported name (pd.concat(...), Redis(...))
     is traced through the file's imports to a dotted path
     (pandas.concat), and through the sample package's own re-exports
     until the path names the library.
  3. That path is followed through the library's re-exports, in the old
     version's files (griffe, static: nothing is imported or run), to
     where the function or class is defined. Where the record names a
     class's __init__, it is looked up along the class's bases.
  4. A method called on an object (df.set_index(...)) needs the object's
     class. It is taken only from a constructor call of a library class:
     the call is chained on it (pd.DataFrame(...).set_index(...)), or the
     object was assigned from it, `df = pd.DataFrame(...)`, by a
     statement that always runs before the call, with no other binding
     of the name that can reach the call. The method is then looked up
     along the class's bases.

A call is "confirmed" only when the path found is exactly the record's
symbol. Everything else is "by hand", with the reason: a name bound in
any other way (a parameter, a loop variable, a second assignment, two
imports that disagree), a receiver that is not a constructor's result,
a path that ends in the sample package's own code. "By hand" means this
script cannot tell; it does not mean the tool was wrong.

What the check does not cover: only the calls in the record are looked
at, so it says nothing about calls the tool missed or reported as
unverified. And the library is read with griffe, as the pipeline reads
it: the check is separate from the tool's code, not from griffe.

The summary also counts the calls for which the library defines another
function or class of the same name (redis has a synchronous and an
asyncio Redis), that is, the calls a match on the name alone would not
settle.

--selftest plants calls in a small made-up library and project, the
right ones beside look-alikes (the project's own function of the same
name, a different class with the same method, a rebound variable), and
passes only if every planted call comes out as written beside it.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import pathlib
import sys
import tempfile
import textwrap

try:
    import griffe
except ImportError:  # pragma: no cover - griffe is the pipeline's dependency
    raise SystemExit("griffe is needed (pip install griffe)")

ROOT = pathlib.Path("data/eval")
RECORD = pathlib.Path("docs/eval/verdicts.csv")
CONFIRMED, BY_HAND = "confirmed", "by hand"
OUT_COLUMNS = ["id", "kind", "sample", "changed", "from", "file", "line",
               "column", "symbol", "status", "route", "detail",
               "same_name_elsewhere"]
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# --------------------------------------------------------------------------
# the library side: one old version, read statically
# --------------------------------------------------------------------------

def final(obj):
    """An alias followed to what it stands for; None if it leads nowhere."""
    for _ in range(30):
        if obj is None or not obj.is_alias:
            return obj
        try:
            obj = obj.target
        except Exception:
            return None
    return None


def method_on(cls, name):
    """The member `name` as the class sees it: its own, else the first one
    along its bases. None when a base that comes earlier lies outside the
    library, because that base could define the name too."""
    cls = final(cls)
    if cls is None or not cls.is_class:
        return None
    try:
        chain = [cls] + list(cls.mro())
    except Exception:
        return None
    for one in chain:
        one = final(one)
        if one is None or not one.is_class:
            return None
        if name in one.members:
            return final(one.members[name])
        try:
            if len(one.resolved_bases) != len(one.bases):
                return None
        except Exception:
            return None
    return None


class Library:
    """One version of a changed package, loaded from a flat directory."""

    def __init__(self, directory: pathlib.Path, root_name: str):
        loader = griffe.GriffeLoader(search_paths=[str(directory)],
                                     allow_inspection=False)
        self.module = loader.load(root_name)
        self.note = ""
        try:
            loader.resolve_aliases(implicit=False, external=False)
        except Exception as exc:
            self.note = f"resolve_aliases: {type(exc).__name__}"

    def resolve(self, dotted: str):
        """A dotted path followed name by name to the object it stands for."""
        parts = dotted.split(".")
        if parts[0] != self.module.name:
            return None
        obj = self.module
        for part in parts[1:]:
            obj = final(obj)
            if obj is None:
                return None
            try:
                if part in obj.members:
                    obj = obj.members[part]
                elif obj.is_class:
                    obj = method_on(obj, part)
                else:
                    return None
            except Exception:
                return None
        return final(obj)


def definitions_by_name(library: Library):
    """Every function and class the library defines, by its last name:
    name -> {(path, "class" or "function")}."""
    out, seen = collections.defaultdict(set), set()

    def walk(obj):
        try:
            members = list(obj.members.values())
        except Exception:
            return
        for member in members:
            if member.is_alias or id(member) in seen:
                continue
            seen.add(id(member))
            if member.is_function or member.is_class:
                kind = "class" if member.is_class else "function"
                out[member.name].add((member.path, kind))
            if member.is_module or member.is_class:
                walk(member)

    walk(library.module)
    return out


def other_definition(symbol: str, by_name) -> str:
    """Another definition with the same last name as the record's symbol
    (for an __init__, another class of the same name): the first by path,
    or "" when the library has none."""
    if symbol.endswith(".__init__"):
        cls = symbol[:-len(".__init__")]
        others = sorted(path + ".__init__"
                        for path, kind in by_name.get(cls.split(".")[-1], ())
                        if kind == "class" and path != cls)
    else:
        others = sorted(path for path, _ in by_name.get(symbol.split(".")[-1], ())
                        if path != symbol)
    return others[0] if others else ""


# --------------------------------------------------------------------------
# the project side: one file, its scopes and its imports
# --------------------------------------------------------------------------

_PARSED: dict = {}


def parse(path: pathlib.Path):
    """A file's syntax tree, every node knowing its parent."""
    if path not in _PARSED:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.parent = None
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        _PARSED[path] = tree
    return _PARSED[path]


def ancestors(node):
    """The nodes around `node`, innermost first, the module last."""
    out = []
    node = getattr(node, "parent", None)
    while node is not None:
        out.append(node)
        node = getattr(node, "parent", None)
    return out


def statement_of(node):
    """The statement an expression is part of."""
    while not isinstance(node, ast.stmt):
        node = node.parent
    return node


def place(node):
    return (node.lineno, node.col_offset)


def package_of(path: pathlib.Path):
    """The dotted package a file belongs to, as a list, by its __init__.py
    files; empty when the file is in no package."""
    parts, folder = [], path.parent
    while (folder / "__init__.py").exists():
        parts.insert(0, folder.name)
        folder = folder.parent
    return parts


def own_nodes(scope):
    """The nodes of one scope, without the insides of a nested function,
    class, lambda or comprehension, whose names bind elsewhere. The nested
    def or class itself is given, for its name, and so is the target of
    an assignment expression inside a comprehension, which binds here."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, COMPS):
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr):
                    yield sub.target
            continue
        if isinstance(node, FUNCS + (ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def bindings(scope, name):
    """How `name` is bound in one scope: (its import statements, every
    other node that binds it, whether it is a parameter)."""
    imports, others, param = [], [], False
    if isinstance(scope, FUNCS + (ast.Lambda,)):
        a = scope.args
        every = a.posonlyargs + a.args + a.kwonlyargs
        every += [x for x in (a.vararg, a.kwarg) if x is not None]
        param = any(x.arg == name for x in every)
        if isinstance(scope, ast.Lambda):
            return imports, others, param
    if isinstance(scope, COMPS):
        for gen in scope.generators:
            for sub in ast.walk(gen.target):
                if isinstance(sub, ast.Name) and sub.id == name:
                    others.append(sub)
        return imports, others, param
    for node in own_nodes(scope):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == name:
                    imports.append(node)
                    break
        elif isinstance(node, FUNCS + (ast.ClassDef,)):
            if node.name == name:
                others.append(node)
        elif isinstance(node, ast.Name):
            if node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
                others.append(node)
        elif isinstance(node, ast.ExceptHandler):
            if node.name == name:
                others.append(node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if name in node.names:
                others.append(node)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name == name:
                others.append(node)
    others.sort(key=place)
    return imports, others, param


def scopes_of(node):
    """The scopes that can bind a name used at `node`, innermost first:
    the comprehensions, lambdas and functions around it, a class only
    when the node sits directly in its body, then the module."""
    out, direct = [], True
    for anc in ancestors(node):
        if isinstance(anc, COMPS + (ast.Lambda,)):
            out.append(anc)
        elif isinstance(anc, FUNCS):
            out.append(anc)
            direct = False
        elif isinstance(anc, ast.ClassDef):
            if direct:
                out.append(anc)
            direct = False
        elif isinstance(anc, ast.Module):
            out.append(anc)
    return out


def binder(node, name):
    """The innermost scope around `node` that binds `name`, and how."""
    for scope in scopes_of(node):
        imports, others, param = bindings(scope, name)
        if imports or others or param:
            return scope, imports, others, param
    return None, [], [], False


def declared_elsewhere(tree, name):
    """Is `name` declared global or nonlocal anywhere in the file?"""
    return any(isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names
               for n in ast.walk(tree))


def import_targets(statements, name, package):
    """The dotted paths `name` stands for after these import statements:
    one, unless they disagree. "?" marks a relative import that cannot be
    placed."""
    targets = set()
    for st in statements:
        for alias in st.names:
            if isinstance(st, ast.Import):
                if alias.asname == name:
                    targets.add(alias.name)
                elif alias.asname is None and alias.name.split(".")[0] == name:
                    targets.add(name)
                continue
            if alias.name == "*" or (alias.asname or alias.name) != name:
                continue
            if st.level:
                if st.level > len(package):
                    targets.add("?")
                    continue
                base = package[:len(package) - (st.level - 1)]
                module = ".".join(base + ([st.module] if st.module else []))
            else:
                module = st.module or "?"
            targets.add(f"{module}.{alias.name}")
    return targets


class Project:
    """One sample package's unpacked source tree."""

    def __init__(self, tree_root: pathlib.Path):
        self.root = tree_root

    def file(self, relative: str) -> pathlib.Path:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{relative} is not in {self.root}")
        return path

    def module_file(self, parts):
        """The project's own file for a dotted module, if it has one."""
        for prefix in ("", "src", "python", "lib"):
            base = self.root / prefix if prefix else self.root
            for cand in (base.joinpath(*parts).with_suffix(".py"),
                         base.joinpath(*parts) / "__init__.py"):
                if cand.is_file():
                    return cand
        tail = "/".join(parts)
        hits = [h for pattern in (f"**/{tail}.py", f"**/{tail}/__init__.py")
                for h in self.root.glob(pattern)]
        return hits[0] if len(hits) == 1 else None

    def to_library(self, dotted, libroot, hops=()):
        """A dotted path followed through the project's own modules until
        it names the library. (path, the project modules passed through),
        or (None, why) when it ends in the project's own code or leaves
        for a third package."""
        if dotted.split(".")[0] == libroot:
            return dotted, hops
        if len(hops) > 6:
            return None, f"{dotted}: more than six re-exports"
        parts = dotted.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            path = self.module_file(parts[:cut])
            if path is None:
                continue
            tree, name = parse(path), parts[cut]
            imports, others, _ = bindings(tree, name)
            if others or not imports or declared_elsewhere(tree, name):
                return None, f"{dotted} is the sample package's own"
            targets = import_targets(imports, name, package_of(path))
            if len(targets) != 1 or "?" in next(iter(targets)):
                return None, f"{dotted} is imported from more than one place"
            onward = ".".join([targets.pop()] + parts[cut + 1:])
            return self.to_library(onward, libroot,
                                   hops + (".".join(parts[:cut]),))
        return None, f"{dotted} is neither the library nor the sample package"


def dotted_callee(func, at, path):
    """The dotted path an expression like pd.concat or Redis stands for,
    by the imports in force where it is used. (None, why) when its first
    name is not plainly an imported one."""
    names, node = [], func
    while isinstance(node, ast.Attribute):
        names.insert(0, node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None, "the callee does not begin with a plain name"
    name = node.id
    scope, imports, others, param = binder(at, name)
    if scope is None:
        return None, f"{name} is not bound in this file"
    if param:
        return None, f"{name} is a parameter"
    if others:
        return None, f"{name} is bound at line {others[0].lineno}, not only imported"
    if declared_elsewhere(parse(path), name):
        return None, f"{name} is declared global or nonlocal in this file"
    targets = import_targets(imports, name, package_of(path))
    if len(targets) != 1:
        return None, f"{name} is imported from {len(targets)} different places"
    target = targets.pop()
    if "?" in target:
        return None, f"the import of {name} cannot be placed"
    return ".".join([target] + names), ""


def runs_before(first, later):
    """Does statement `first` always run before statement `later`? True
    when `later` is, or lies inside, a statement that follows `first` in
    the same block."""
    block = None
    for field in ("body", "orelse", "finalbody"):
        got = getattr(first.parent, field, None)
        if isinstance(got, list) and any(s is first for s in got):
            block = got
    if block is None:
        return False
    index = next(i for i, s in enumerate(block) if s is first)
    around = [later] + ancestors(later)
    return any(any(s is a for a in around) for s in block[index + 1:])


def assignment_for(call, name, path):
    """The statement `name = <call>` that gives `name` its value at this
    call, when there is exactly one it can be; else (None, why)."""
    scope, imports, others, param = binder(call, name)
    if scope is None:
        return None, f"{name} is not bound in this file"
    if param:
        return None, f"{name} is a parameter"
    if imports:
        return None, f"{name} is an imported name"
    if isinstance(scope, COMPS + (ast.Lambda, ast.ClassDef)):
        return None, f"{name} is bound by a comprehension or a class body"
    if declared_elsewhere(parse(path), name):
        return None, f"{name} is declared global or nonlocal in this file"
    plain = []
    for node in others:
        st = getattr(node, "parent", None)
        if isinstance(st, ast.Assign) and any(t is node for t in st.targets):
            if isinstance(st.value, ast.Call):
                plain.append(st)
        elif isinstance(st, ast.AnnAssign) and st.target is node:
            if isinstance(st.value, ast.Call):
                plain.append(st)
    home = next(s for s in scopes_of(call) if not isinstance(s, COMPS))
    if home is not scope:
        # the name belongs to an outer scope: the call can run at any time
        if len(others) != 1 or len(plain) != 1:
            return None, (f"{name} belongs to an outer scope and is bound "
                          f"{len(others)} times there")
        return plain[0], ""
    here = statement_of(call)
    before = [st for st in plain
              if (st.end_lineno, st.end_col_offset) <= place(here)]
    if not before:
        return None, f"no statement `{name} = <call>` before the call"
    chosen = max(before, key=place)
    if not runs_before(chosen, here):
        return None, (f"the assignment at line {chosen.lineno} is in a "
                      "branch the call is not in")
    for node in others:
        if statement_of(node) is chosen:
            continue
        if place(chosen) < place(node) < place(here):
            return None, f"{name} is bound again at line {node.lineno}"
    around_chosen = ancestors(chosen)
    for loop in ancestors(call):
        if loop is scope:
            break
        if not isinstance(loop, LOOPS) or any(loop is a for a in around_chosen):
            continue
        for node in others:
            if any(loop is a for a in ancestors(node)):
                return None, (f"{name} is bound again at line {node.lineno}, "
                              f"inside the loop at line {loop.lineno}")
    return chosen, ""


# --------------------------------------------------------------------------
# one call
# --------------------------------------------------------------------------

def find_call(tree, line, column, leaf):
    """The call the record points at: the one that begins at its line and
    column; among several beginning there (a chain), the one whose called
    name is the record's."""
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def called(c):
        if isinstance(c.func, ast.Attribute):
            return c.func.attr
        return c.func.id if isinstance(c.func, ast.Name) else None

    at = [c for c in calls if c.lineno == line and c.col_offset + 1 == column]
    if len(at) > 1:
        at = [c for c in at if called(c) == leaf]
    if len(at) == 1:
        return at[0], "at the line and column"
    on_line = [c for c in calls if c.lineno == line and called(c) == leaf]
    if len(on_line) == 1:
        return on_line[0], "by its name on the line"
    return None, f"{len(at)} call(s) begin at {line}:{column}"


def check(row, project: Project, library: Library):
    """(status, route, detail) for one row of the record."""
    symbol = row["symbol"]
    libroot = symbol.split(".")[0]
    is_init = symbol.endswith(".__init__")
    leaf = symbol.split(".")[-2] if is_init else symbol.split(".")[-1]
    path = project.file(row["file"])
    tree = parse(path)
    call, found = find_call(tree, int(row["line"]), int(row["column"]), leaf)
    if call is None:
        return BY_HAND, "", found

    def library_object(func, at):
        """The library object an imported-name expression stands for."""
        dotted, why = dotted_callee(func, at, path)
        if dotted is None:
            return None, why, None
        full, hops = project.to_library(dotted, libroot)
        if full is None:
            return None, hops, dotted
        obj = library.resolve(full)
        if obj is None:
            return None, f"{full} cannot be followed in the library", dotted
        via = "".join(f" (through {h})" for h in hops)
        return obj, f"{dotted}{via}", dotted

    # a call written through an imported name
    obj, text, dotted = library_object(call.func, call)
    if obj is not None:
        route = ("an imported name" if dotted.split(".")[0] == libroot
                 else "an imported name, re-exported by the sample package")
        if is_init:
            if not obj.is_class:
                return BY_HAND, route, f"{text} is not a class"
            init = method_on(obj, "__init__")
            got = init.path if init is not None else None
        else:
            got = obj.path
        status = CONFIRMED if got == symbol else BY_HAND
        return status, route, f"{text} -> {got}"
    if dotted is not None or not isinstance(call.func, ast.Attribute):
        return BY_HAND, "", text

    # a method called on an object
    receiver, method = call.func.value, call.func.attr
    if isinstance(receiver, ast.Call):
        made, route, how = receiver, "a method, chained on the constructor", ""
    elif isinstance(receiver, ast.Name):
        st, why = assignment_for(call, receiver.id, path)
        if st is None:
            return BY_HAND, "", why
        made, route = st.value, "a method, on an object assigned from the constructor"
        how = f"{receiver.id} assigned at line {st.lineno}: "
    else:
        return BY_HAND, "", "the receiver is neither a name nor a constructor call"
    cls, text, _ = library_object(made.func, made)
    if cls is None:
        return BY_HAND, "", f"{how}{text}"
    if not cls.is_class:
        return BY_HAND, "", f"{how}{text} is not a class"
    member = method_on(cls, method)
    got = member.path if member is not None else None
    status = CONFIRMED if got == symbol else BY_HAND
    return status, route, f"{how}{text} is {cls.path}; .{method} -> {got}"


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

def tree_root(root: pathlib.Path, sample: str) -> pathlib.Path:
    """The one unpacked tree of a sample package under <root>/trees/."""
    found = [p for p in (root / "trees" / sample / "src").iterdir() if p.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"{len(found)} trees under trees/{sample}/src")
    return found[0]


def run(rows, root: pathlib.Path):
    """Every row checked; a list of result rows in the record's order."""
    projects, libraries, names, out = {}, {}, {}, []
    for row in rows:
        result = {k: row[k] for k in OUT_COLUMNS if k in row}
        result["same_name_elsewhere"] = ""
        try:
            sample = row["sample"]
            if sample not in projects:
                projects[sample] = Project(tree_root(root, sample))
            key = (row["changed"], row["from"], row["symbol"].split(".")[0])
            if key not in libraries:
                folder = root / "old" / f"{key[0]}-{key[1]}"
                if not folder.is_dir():
                    raise FileNotFoundError(f"{folder} is missing")
                libraries[key] = Library(folder, key[2])
                names[key] = definitions_by_name(libraries[key])
            result["same_name_elsewhere"] = other_definition(row["symbol"], names[key])
            status, route, detail = check(row, projects[sample], libraries[key])
        except Exception as exc:
            status, route = BY_HAND, ""
            detail = f"the check stopped: {type(exc).__name__}: {exc}"
        result.update(status=status, route=route, detail=detail)
        out.append(result)
    return out


def summary(results) -> str:
    n = len(results)
    ok = [r for r in results if r["status"] == CONFIRMED]
    lines = [f"{n} calls in the record",
             f"  confirmed  {len(ok):4d}   the path found is the record's symbol"]
    for route, k in collections.Counter(r["route"] for r in ok).most_common():
        lines.append(f"     {k:4d}   {route}")
    rest = [r for r in results if r["status"] != CONFIRMED]
    lines.append(f"  by hand    {len(rest):4d}   this script cannot tell")
    for r in rest:
        lines.append(f"     #{r['id']} {r['sample']}/{r['file']}:{r['line']}:"
                     f"{r['column']}  {r['symbol']}")
        lines.append(f"          {r['detail']}")
    twins = sum(1 for r in results if r["same_name_elsewhere"])
    lines.append(f"for {twins} of the {n} calls the library defines another function or "
                 "class of the same name")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the self-test: planted calls, the right ones beside look-alikes
# --------------------------------------------------------------------------

_LIB = {
    "lib/__init__.py": """
        from .core import Frame, Series, concat
        from .client import Client
        from . import aio
        """,
    "lib/core.py": """
        class Base:
            def astype(self, dtype):
                return self


        class Frame(Base):
            def __init__(self, data=None):
                self.data = data

            def set_index(self, keys):
                return self

            def groupby(self, by=None):
                return self


        class Series(Base):
            def __init__(self, data=None):
                self.data = data

            def groupby(self, by=None):
                return self


        def concat(objs, axis=0):
            return objs
        """,
    "lib/client.py": """
        class Client:
            def __init__(self, host="h", port=1):
                self.host = host
        """,
    "lib/aio/__init__.py": """
        from .client import Client
        """,
    "lib/aio/client.py": """
        class Client:
            def __init__(self, host="h", port=1):
                self.host = host
        """,
}

_PROJECT = {
    "proj/__init__.py": "",
    "proj/own.py": """
        class Frame:
            def set_index(self, keys):
                return self


        class Client:
            def __init__(self, host=None):
                self.host = host


        class Namespace:
            def concat(self, objs):
                return objs


        def concat(objs):
            return objs
        """,
    "proj/compat.py": """
        from lib import concat as lib_concat
        from .own import concat as own_concat
        """,
    # every planted call is a statement of its own, marked  # <id>
    "proj/cases.py": """
        import lib as lb
        from lib import Client, aio
        from lib import concat as concat2
        from .compat import lib_concat, own_concat
        from .own import Client as OwnClient
        from .own import Frame as OwnFrame
        from .own import Namespace, concat

        try:
            from lib import concat as either
        except ImportError:
            from .own import concat as either

        SHARED = lb.Frame()
        TWICE = lb.Frame()
        TWICE = OwnFrame()


        def right(a, b, n):
            lb.concat([a, b])  # T1
            concat2([a, b])  # T2
            lib_concat([a, b])  # T3
            df = lb.Frame()
            df.groupby("a")  # T4
            df.astype(int)  # T5
            Client(host="x")  # T6
            lb.Frame({}).set_index("a")  # T7
            SHARED.set_index("a")  # T8
            for _ in range(n):
                again = lb.Frame()
                again.set_index("a")  # T9
                again = OwnFrame()


        def own_function(a, b):
            concat([a, b])  # N1


        def own_class():
            df = OwnFrame()
            df.set_index("a")  # N2


        def other_class():
            s = lb.Series()
            s.groupby("a")  # N3


        def parameter(df):
            df.set_index("a")  # N4


        def rebound():
            df = lb.Frame()
            df = OwnFrame()
            df.set_index("a")  # N5


        def rebound_in_a_branch(flag):
            df = lb.Frame()
            if flag:
                df = OwnFrame()
            df.set_index("a")  # N6


        def library_only_in_a_branch(flag):
            df = OwnFrame()
            if flag:
                df = lb.Frame()
            df.set_index("a")  # N7


        def rebound_in_the_loop(n):
            df = lb.Frame()
            for _ in range(n):
                df.set_index("a")  # N8
                df = OwnFrame()


        def other_module():
            aio.Client(host="x")  # N9


        def alias_shadowed():
            lb = Namespace()
            lb.concat([1, 2])  # N10


        def own_through_compat(a, b):
            own_concat([a, b])  # N11


        def loop_variable(frames):
            for SHARED in frames:
                SHARED.set_index("a")  # N13


        def bound_twice_outside():
            TWICE.set_index("a")  # N14


        def two_imports(a, b):
            either([a, b])  # N15


        def own_client():
            OwnClient(host="x")  # N16


        def unpacked():
            df, other = lb.Frame(), OwnFrame()
            df.set_index("a")  # N17


        def not_a_constructor(a, b):
            df = lb.concat([a, b])
            df.set_index("a")  # N18


        def bound_in_try():
            try:
                df = lb.Frame()
            except Exception:
                df = OwnFrame()
            df.set_index("a")  # N19
        """,
}

# id: (the symbol the record would name, what the check must say).
# Written before the self-test was first run.
_EXPECTED = {
    "T1": ("lib.core.concat", CONFIRMED),            # through `import lib as lb`
    "T2": ("lib.core.concat", CONFIRMED),            # `from lib import concat as concat2`
    "T3": ("lib.core.concat", CONFIRMED),            # re-exported by the project's compat module
    "T4": ("lib.core.Frame.groupby", CONFIRMED),     # object assigned from the constructor
    "T5": ("lib.core.Base.astype", CONFIRMED),       # the method is a base class's
    "T6": ("lib.client.Client.__init__", CONFIRMED),
    "T7": ("lib.core.Frame.set_index", CONFIRMED),   # chained on the constructor
    "T8": ("lib.core.Frame.set_index", CONFIRMED),   # bound once, at module level
    "T9": ("lib.core.Frame.set_index", CONFIRMED),   # assigned afresh in every turn of the loop
    "N1": ("lib.core.concat", BY_HAND),              # the project's own concat
    "N2": ("lib.core.Frame.set_index", BY_HAND),     # the project's own Frame
    "N3": ("lib.core.Frame.groupby", BY_HAND),       # a Series, not a Frame
    "N4": ("lib.core.Frame.set_index", BY_HAND),     # a parameter: class unknown
    "N5": ("lib.core.Frame.set_index", BY_HAND),     # assigned again before the call
    "N6": ("lib.core.Frame.set_index", BY_HAND),     # assigned again in a branch
    "N7": ("lib.core.Frame.set_index", BY_HAND),     # the library class only in a branch
    "N8": ("lib.core.Frame.set_index", BY_HAND),     # assigned again later in the loop
    "N9": ("lib.client.Client.__init__", BY_HAND),   # it is lib.aio.client.Client
    "N10": ("lib.core.concat", BY_HAND),             # lb is a local object here
    "N11": ("lib.core.concat", BY_HAND),             # compat hands on the project's own
    "N12": ("lib.core.concat", BY_HAND),             # no call at the recorded place
    "N13": ("lib.core.Frame.set_index", BY_HAND),    # a loop variable of the same name
    "N14": ("lib.core.Frame.set_index", BY_HAND),    # bound twice at module level
    "N15": ("lib.core.concat", BY_HAND),             # two imports that disagree
    "N16": ("lib.client.Client.__init__", BY_HAND),  # the project's own Client
    "N17": ("lib.core.Frame.set_index", BY_HAND),    # tuple assignment, not `name = <call>`
    "N18": ("lib.core.Frame.set_index", BY_HAND),    # assigned from a function's result
    "N19": ("lib.core.Frame.set_index", BY_HAND),    # try and except bind it differently
}


def selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for files, base in ((_LIB, root / "old" / "lib-1.0"),
                            (_PROJECT, root / "trees" / "proj" / "src" / "proj-1.0")):
            for rel, text in files.items():
                target = base / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        cases = (root / "trees/proj/src/proj-1.0/proj/cases.py").read_text(
            encoding="utf-8").split("\n")
        where = {}
        for number, text in enumerate(cases, start=1):
            if "  # " in text:
                mark = text.rsplit("  # ", 1)[1].strip()
                where[mark] = (number, len(text) - len(text.lstrip()) + 1)
        def_line = next(i for i, t in enumerate(cases, 1) if t.startswith("def right"))
        where["N12"] = (def_line, 1)
        rows = []
        for mark, (symbol, _) in _EXPECTED.items():
            line, column = where[mark]
            rows.append({"id": mark, "kind": "planted", "sample": "proj",
                         "changed": "lib", "from": "1.0", "file": "proj/cases.py",
                         "line": str(line), "column": str(column), "symbol": symbol})
        results = {r["id"]: r for r in run(rows, root)}
    failed = 0
    for mark, (symbol, want) in _EXPECTED.items():
        got = results[mark]
        good = got["status"] == want
        failed += not good
        print(f"{'ok  ' if good else 'FAIL'} {mark:4s} {want:10s} {symbol}\n"
              f"          {got['detail']}")
    print(f"{len(_EXPECTED)} checks, {failed} failed")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=ROOT,
                    help="the evaluation's folder (trees/ and old/ in it)")
    ap.add_argument("--record", type=pathlib.Path, default=RECORD)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: <root>/locate_check.csv")
    ap.add_argument("--selftest", action="store_true",
                    help="run the planted calls and stop")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    with args.record.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    results = run(rows, args.root)
    out = args.out or args.root / "locate_check.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(summary(results))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
