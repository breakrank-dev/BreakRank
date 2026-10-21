"""
Locate (NOTES §38.3): which lines of a project use each changed name.

Static only. Every `.py` under the project is parsed with libcst; nothing
is imported from the project or from the package (§17's rule). libcst's
qualified-name metadata resolves the project's imports (`import pkg`,
`import pkg.mod as m`, `from pkg.mod import name as alias`) into the
dotted path each name stands for, and every call, attribute, import,
base class and argument is matched against the changed symbols' paths:
the definition path and every export path the ingest recorded, since
downstream code imports the short one (`_pytest.fixtures.fixture` is
written `pytest.fixture`, so every export path's root counts as the
package's). A path the bundle does not list is resolved through the
installed old version when it was read (an alias `alphalib.Client` to
its definition `alphalib.core.Client`, a method to the class that
defines it). A removed module is matched in every path that passes
through it (`import pkg.gone`, `from pkg.gone import x`, `pkg.gone.f()`).

What a hit is, and how sure:

  resolved     the code names a changed path through an import the tool
               followed: a call to the function (or to the class, for a
               change on its `__init__`); a method called on an instance
               the tool saw made, by `name = Class(...)`, `with Class(...)
               as name` or `name: Class = Class(...)`, in the same
               function, an enclosing one or the module, or by `self.x =
               Class(...)` anywhere in the class (a class attribute
               `x = Class()` counts too), the defining class found through
               the old version's bases so a change recorded on a mixin
               matches `Redis(...).set(...)`; a read of a constant; a
               subclass or `isinstance` check for a class whose base was
               removed; an import of a removed name or module.
  referenced   the changed object is named but not called here, so a
               change to its parameters cannot be judged: passed as an
               argument (`partial(f, 1, 2)`), used as a base class, or
               reached by `super().__init__(...)` in a project subclass,
               which the tool does not follow.
  unverified   a method of a changed name called on an object whose class
               the tool does not know: a parameter, a loop variable, a
               result of a function (the package's own `connect(...)` and
               `Redis.from_url(...)` included: return types are not
               inferred), an element of a tuple assignment, `self.set(...)`
               in a project class, a name from the project's own modules;
               or known only by name when the old version was not read.
               Listed, never rewritten; listed in the files that import
               the package, since in a file that never names it
               `d.get(k)` is a dict's.

A name rebound shadows what it hid: a parameter, loop variable, `except
... as`, lambda parameter or local `def` named like a module-level
instance is unknown inside its scope, not that instance, and a local
`def` or `class` named like an imported name hides the import. A changed
constant used as a receiver (`pkg.VERSION.split(".")`) is a read of it.

Without the old version the tool takes nothing for a class that the
bundle does not show to be one (a class with changed `__init__` rows, or
a removed base), so `Base.send(c, ...)` is recorded as a call with no
receiver and no self to offset, and `b = Base(); b.send()` is
unverified; R1 needs the old version anyway (§38.4). A name a package
binds by assignment (`StrictRedis = Redis`) is an attribute to griffe,
not the class: a call to it is no constructor hit, and its instance is
unknown.

Not reported: a method called on an object made by another package's
call (`requests.Session().send(...)`, `s = requests.get(...); s.send()`)
or bound to a literal, which is that package's or Python's; a name only
annotated (`def f(c: Client)`); an import of a name whose parameters
changed, since the call is where that shows. Not followed, and said in
the report: a project module that re-exports the package's names (`from
myapp.clients import Client`), so a file that reaches the package only
through the project's own modules is counted, not read for them. `from
pkg import *` is reported as a file the tool cannot see through;
`getattr`, strings and `importlib` are invisible and the report says so
once. A file libcst cannot parse or analyse, or not in UTF-8, is
reported and skipped. Environments and build output are not read
(SKIP_ANYWHERE at any depth, SKIP_AT_TOP directly under the project, any
directory holding a `pyvenv.cfg`), and the report names them; a file
whose text never mentions the package's import roots is read but not
parsed, since no hit is possible in it.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re

import libcst as cst
from libcst.metadata import (MetadataWrapper, PositionProvider,
                             QualifiedNameProvider)

from fixer.bundle import Change
from fixer.oldapi import OldApi

SKIP_ANYWHERE = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv",
                 "node_modules", ".tox", ".nox", ".mypy_cache",
                 ".pytest_cache", ".ruff_cache", "site-packages", ".eggs",
                 ".idea", ".vscode"}
SKIP_AT_TOP = {"env", ".env", "build", "dist"}
# Never code, so not worth a line in the report; the rest of what is not
# entered (environments, build output) is named there.
SKIP_SILENT = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache",
               ".pytest_cache", ".ruff_cache", ".eggs", ".idea", ".vscode"}

# A change about the object itself is settled by a reference to it; a
# change about its parameters or its result needs a call.
REFERENCE_KINDS = {"OBJECT_REMOVED", "OBJECT_CHANGED_KIND",
                   "CLASS_REMOVED_BASE", "ATTRIBUTE_CHANGED_VALUE"}
# An import line, or a path passing through a module, is a hit only where
# the name is gone or is no longer a class or function.
IMPORT_KINDS = {"OBJECT_REMOVED", "OBJECT_CHANGED_KIND"}
# A changed object the code names on the way to something else
# (`pkg.gone.f`, `pkg.VERSION.split`): a removed module or object, or a
# constant read as a receiver.
PREFIX_KINDS = IMPORT_KINDS | {"ATTRIBUTE_CHANGED_VALUE"}

RESOLVED, REFERENCED, UNVERIFIED = "resolved", "referenced", "unverified"
FOREIGN = "<another package's object>"
UNKNOWN = "<unknown>"

# A value that is never an instance of the package's classes: a literal,
# a comprehension, a string, a number, a lambda, a comparison.
LITERALS = (cst.Dict, cst.List, cst.Set, cst.Tuple, cst.DictComp,
            cst.ListComp, cst.SetComp, cst.GeneratorExp, cst.SimpleString,
            cst.ConcatenatedString, cst.FormattedString, cst.Integer,
            cst.Float, cst.Imaginary, cst.Lambda, cst.Comparison,
            cst.BooleanOperation)
COMPREHENSIONS = (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)
NOT_INSTANCE_METHOD = {"classmethod", "staticmethod"}


@dataclasses.dataclass(frozen=True)
class Hit:
    file: str
    line: int
    column: int                     # 1-based, as editors show it
    source: str
    # the dotted path the code names; "?.name" when the receiver is unknown
    path: str
    change_ids: tuple[int, ...]
    confidence: str                 # resolved | referenced | unverified
    # call | method | attribute | import | subclass | argument | super
    use: str
    # a method or class call: the class, by its defining path when the old
    # version was read, else as the code names it
    receiver: str | None = None
    explicit_self: bool = False     # Class.method(obj, ...) form
    positional: int = 0             # arguments passed by position
    keywords: tuple[str, ...] = ()  # arguments passed by name
    star_args: bool = False         # any *args or **kwargs at the call


@dataclasses.dataclass(frozen=True)
class Note:
    file: str
    text: str


@dataclasses.dataclass
class Located:
    hits: list[Hit]
    notes: list[Note]
    files_read: int
    files_parsed: int = 0       # those that mention an import root
    # those that reach the package only through the project's own
    # modules, which are not followed
    files_via_project: int = 0
    roots: tuple[str, ...] = ()
    skipped_dirs: tuple[str, ...] = ()

    def by_change(self) -> dict[int, list[Hit]]:
        out: dict[int, list[Hit]] = {}
        for h in self.hits:
            for cid in h.change_ids:
                out.setdefault(cid, []).append(h)
        return out

    def counts(self) -> dict[str, int]:
        out = {RESOLVED: 0, REFERENCED: 0, UNVERIFIED: 0}
        for h in self.hits:
            out[h.confidence] += 1
        return out

    def lines(self) -> int:
        return len({(h.file, h.line) for h in self.hits})


class Index:
    """The changed names of one upgrade, by every path code can write."""

    def __init__(self, changes: list[Change], api: OldApi | None = None,
                 project_roots: set[str] | None = None):
        self.api = api if api is not None and api.loaded else None
        self.project_roots = set(project_roots or ())
        self.changes = {c.id: c for c in changes}
        self.by_path: dict[str, set[int]] = {}
        self.ctor: dict[str, set[int]] = {}       # class path -> __init__ ids
        self.classes: set[str] = set()             # paths known to be classes
        self.by_leaf: dict[str, set[int]] = {}    # method name -> ids
        self.roots: set[str] = set()
        for c in changes:
            for p in c.paths:
                self.roots.add(p.split(".")[0])
                self.by_path.setdefault(p, set()).add(c.id)
                if p.endswith(".__init__"):
                    cls = p[:-len(".__init__")]
                    self.ctor.setdefault(cls, set()).add(c.id)
                    self.classes.add(cls)
                if c.kind == "CLASS_REMOVED_BASE":
                    self.classes.add(p)
            method = self._is_method(c.symbol)
            if method:
                self.by_leaf.setdefault(c.symbol.rsplit(".", 1)[1],
                                        set()).add(c.id)
            if method and self.api is not None:
                self.classes.update(p.rsplit(".", 1)[0] for p in c.paths
                                    if "." in p)

    def _is_method(self, symbol: str) -> bool:
        """Whether the symbol is a member of a class: asked of the old
        version when it was read; else guessed from the parent's spelling
        (a class name starts with a capital), which decides only which
        names an unknown receiver's call is checked against."""
        parent, _, _leaf = symbol.rpartition(".")
        if "." not in parent:
            return False
        if self.api is not None:
            obj = self.api.lookup(parent)
            if obj is not None:
                return bool(getattr(obj, "is_class", False))
        return parent.rsplit(".", 1)[-1][:1].isupper()

    def ours(self, path: str) -> bool:
        return path.split(".")[0] in self.roots

    def projects(self, path: str) -> bool:
        """A relative import, or an absolute one of the project's own
        top-level package or module."""
        return path.startswith(".") or path.split(".")[0] in self.project_roots

    def direct(self, path: str) -> set[int]:
        """Changes on exactly this path, by the bundle's paths or through
        the old version's alias resolution."""
        found = set(self.by_path.get(path, ()))
        if not found and self.api is not None and self.ours(path):
            canon = self.api.canonical(path)
            if canon:
                found = set(self.by_path.get(canon, ()))
        return found

    def constructor(self, class_path: str) -> set[int]:
        found = set(self.ctor.get(class_path, ()))
        if not found and self.api is not None and self.ours(class_path):
            canon = self.api.canonical(class_path)
            if canon:
                found = set(self.ctor.get(canon, ()))
        return found

    def canonical(self, path: str) -> str:
        """The defining path when the old version was read, else the path."""
        if self.api is not None and self.ours(path):
            return self.api.canonical(path) or path
        return path

    def is_class(self, path: str) -> bool:
        if path in self.classes:
            return True
        if self.api is not None and self.ours(path):
            obj = self.api.lookup(path)
            return obj is not None and bool(getattr(obj, "is_class", False))
        return False

    def method(self, class_path: str, member: str) -> tuple[set[int], str]:
        """Changes on `member` as the class or its bases define it, and the
        defining class's path (the class as written when the old version
        was not read)."""
        owner = self.canonical(class_path)
        found = self.direct(f"{class_path}.{member}")
        if found:
            canon = self.canonical(f"{class_path}.{member}")
            if "." in canon:
                owner = canon.rsplit(".", 1)[0]
            return found, owner
        if self.api is not None:
            defining = self.api.defining_class(class_path, member)
            if defining:
                return (set(self.by_path.get(f"{defining}.{member}", ())),
                        defining)
        return set(), owner

    def instance_method(self, path: str) -> bool:
        """False for a classmethod or staticmethod, when the old version
        says so; True otherwise."""
        if self.api is None:
            return True
        return not (self.api.labels(path) & NOT_INSTANCE_METHOD)

    def leaf(self, member: str) -> set[int]:
        return set(self.by_leaf.get(member, ()))

    def removed_prefix(self, path: str) -> tuple[set[int], str]:
        """A removed (or re-kinded) module or object, or a changed constant,
        that the path passes through: `pkg.gone.f` for a change on
        `pkg.gone`, `pkg.LIMIT.bit_length` for one on `pkg.LIMIT`. The
        longest such prefix, with its ids; empty when none."""
        parts = path.split(".")
        for n in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:n])
            ids, _ = self.settled(self.direct(prefix), PREFIX_KINDS)
            if ids:
                return ids, prefix
        return set(), ""

    def settled(self, ids: set[int], kinds: set[str]) -> tuple[set[int],
                                                               set[int]]:
        """Split ids into those a reference settles and those it does not."""
        yes = {i for i in ids if self.changes[i].kind in kinds}
        return yes, ids - yes


def _dotted(node) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr.value}" if base else None
    return None


def _target_names(node) -> list[str]:
    """The plain names an assignment target binds (`a`, `(a, b)`, `[a, *b]`,
    `a.x` is not one)."""
    if isinstance(node, cst.Name):
        return [node.value]
    if isinstance(node, (cst.Tuple, cst.List)):
        out: list[str] = []
        for el in node.elements:
            out += _target_names(el.value)
        return out
    if isinstance(node, cst.StarredElement):
        return _target_names(node.value)
    return []


def _param_names(params: cst.Parameters) -> list[str]:
    out = [p.name.value for p in params.posonly_params]
    out += [p.name.value for p in params.params]
    out += [p.name.value for p in params.kwonly_params]
    if isinstance(params.star_arg, cst.Param):
        out.append(params.star_arg.name.value)
    if params.star_kwarg is not None:
        out.append(params.star_kwarg.name.value)
    return out


def _call_shape(call: cst.Call) -> dict:
    positional, keywords, star = 0, [], False
    for a in call.args:
        if a.star:
            star = True
        elif a.keyword is not None:
            keywords.append(a.keyword.value)
        else:
            positional += 1
    return dict(positional=positional, keywords=tuple(keywords),
                star_args=star)


class _Names(cst.CSTVisitor):
    """Shared: the qualified names libcst resolves, and what a call makes."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, index: Index):
        self.index = index

    def _qualified(self, node) -> list:
        value = self.get_metadata(QualifiedNameProvider, node, None)
        if value is None:
            return []
        value = value() if callable(value) else value
        return list(value)

    def _names(self, node) -> list[str]:
        """The imported paths a node stands for (a LOCAL name is a project
        variable, a BUILTIN is Python's)."""
        return sorted(n.name for n in self._qualified(node)
                      if n.source.name == "IMPORT")

    def _class_of_call(self, call: cst.Call) -> str:
        """The class a call makes an instance of: its path when the callee
        is a known class, FOREIGN when the callee is another package's,
        UNKNOWN when the tool cannot tell (a factory, a local)."""
        names = self._names(call.func)
        for q in names:
            if self.index.is_class(q):
                return q
        if self._foreign(names):
            return FOREIGN
        return UNKNOWN

    def _foreign(self, names: list[str]) -> bool:
        """Whether resolved names all belong to another package: not the
        upgraded one's, and not the project's own modules, which the tool
        does not follow and treats as unknown."""
        return bool(names) and not any(
            self.index.ours(q) or self.index.projects(q) for q in names)

    def _bind(self, scope: dict[str, str], key: str, value) -> None:
        """Record what a name is bound to: the class a call makes, another
        name's binding in the same scope, FOREIGN for a literal, UNKNOWN
        for anything else (so it shadows an outer binding)."""
        if isinstance(value, cst.Call):
            scope[key] = self._class_of_call(value)
        elif isinstance(value, cst.Name):
            scope[key] = scope.get(value.value, UNKNOWN)
        elif isinstance(value, LITERALS):
            scope[key] = FOREIGN
        else:
            scope[key] = UNKNOWN

    def _bind_target(self, scope: dict[str, str], target, value) -> None:
        """An assignment target: one name takes the value's binding, the
        names of a tuple are unknown (not tracked), `self.x` is kept under
        that key, anything else binds nothing."""
        if isinstance(target, cst.Name):
            self._bind(scope, target.value, value)
        elif (isinstance(target, cst.Attribute)
              and isinstance(target.value, cst.Name)
              and target.value.value == "self"):
            self._bind(scope, f"self.{target.attr.value}", value)
        else:
            for name in _target_names(target):
                scope[name] = UNKNOWN


class _Prescan(_Names):
    """Before the file is read in order: the instances made at module level
    (a function may use one assigned below it) and the `self.x = Class()`
    assignments of each class (a method may use one another sets)."""

    def __init__(self, index: Index):
        super().__init__(index)
        self.module: dict[str, str] = {}
        self.classes: dict[int, dict[str, str]] = {}
        self.class_stack: list[int] = []
        self.depth = 0

    def visit_FunctionDef(self, node) -> None:
        self.depth += 1

    def leave_FunctionDef(self, node) -> None:
        self.depth -= 1

    def visit_ClassDef(self, node) -> None:
        self.class_stack.append(id(node))
        self.classes[id(node)] = {}
        self.depth += 1

    def leave_ClassDef(self, node) -> None:
        self.class_stack.pop()
        self.depth -= 1

    def _record(self, target, value) -> None:
        if self.depth == 0:
            self._bind_target(self.module, target, value)
        elif (self.class_stack and isinstance(target, cst.Attribute)
              and isinstance(target.value, cst.Name)
              and target.value.value == "self"):
            self._bind_target(self.classes[self.class_stack[-1]], target,
                              value)

    def visit_Assign(self, node: cst.Assign) -> None:
        for t in node.targets:
            self._record(t.target, node.value)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if node.value is not None:
            self._record(node.target, node.value)

    def visit_With(self, node: cst.With) -> None:
        for item in node.items:
            if item.asname is not None:
                self._record(item.asname.name, item.item)


class _Finder(_Names):
    """The file in order: hits are kept against their nodes and given
    their positions afterwards, only when there is one (positions are
    the costly metadata, and most files have no hit)."""

    def __init__(self, index: Index, pre: _Prescan):
        super().__init__(index)
        self.pre = pre
        self.pending: list[tuple] = []    # (node, path, ids, conf, use, extra)
        self.star_imports: list[tuple] = []           # (node, module)
        self.imports_package = False
        self.scopes: list[dict[str, str]] = [dict(pre.module)]
        self.kinds: list[str] = ["module"]    # what each scope belongs to
        self.class_pre: list[dict[str, str]] = []   # the prescan's, per class
        self.class_stack: list[tuple[str, list[str]]] = []   # (name, bases)
        self.skip: set[int] = set()       # nodes of a call's callee chain
        self.arg_nodes: set[int] = set()  # argument values
        self.base_nodes: set[int] = set()
        self.annotation = 0

    # ------------------------------------------------------------ helpers

    def _add(self, node, path: str, ids: set[int], confidence: str,
             use: str, **extra) -> None:
        if ids:
            self.pending.append((node, path, set(ids), confidence, use,
                                 extra))

    def _add_reference(self, node, path: str, ids: set[int], use: str,
                       kinds: set[str] = REFERENCE_KINDS) -> None:
        """A reference that is not a call: resolved for the kinds it
        settles, referenced for the rest."""
        yes, no = self.index.settled(ids, kinds)
        self._add(node, path, yes, RESOLVED, use)
        self._add(node, path, no, REFERENCED, use)

    def _lookup_var(self, name: str) -> str | None:
        """The nearest binding of a name: a class path, FOREIGN, or None
        when unknown or unbound. A rebinding shadows what it hid; a class
        body's names are visible in the body alone, not in its methods."""
        last = len(self.scopes) - 1
        for i in range(last, -1, -1):
            if self.kinds[i] == "class" and i != last:
                continue
            if name in self.scopes[i]:
                found = self.scopes[i][name]
                return None if found == UNKNOWN else found
        return None

    def _lookup_self(self, attr: str) -> str | None:
        """`self.attr` as the nearest enclosing class binds it."""
        for scope, kind in zip(reversed(self.scopes), reversed(self.kinds)):
            if kind == "class":
                found = scope.get(f"self.{attr}", UNKNOWN)
                return None if found == UNKNOWN else found
        return None

    def _bound(self, name: str) -> bool:
        """Whether the project binds the name in a scope the tool can see:
        then libcst's import for it is hidden, whatever it is bound to."""
        last = len(self.scopes) - 1
        for i in range(last, -1, -1):
            if self.kinds[i] == "class" and i != last:
                continue
            if name in self.scopes[i]:
                return True
        return False

    def _names(self, node) -> list[str]:
        """libcst's imported paths for the node, unless the project rebinds
        the name at its root (`def Client(...)`, `class Pool`, a parameter
        called `alphalib`): then the import is hidden here."""
        base = node
        while isinstance(base, cst.Attribute):
            base = base.value
        if isinstance(base, cst.Name) and self._bound(base.value):
            return []
        return super()._names(node)

    def _receiver(self, value: cst.BaseExpression) -> str | None:
        """The class an expression is an instance of, when the tool knows:
        a variable bound in a scope it can see, or `self.x` bound in this
        class (a call's result, `Client().send()`, libcst names itself)."""
        if isinstance(value, cst.Name):
            return self._lookup_var(value.value)
        if (isinstance(value, cst.Attribute)
                and isinstance(value.value, cst.Name)
                and value.value.value == "self" and self.class_stack):
            return self._lookup_self(value.attr.value)
        return None

    def _mark_chain(self, node) -> None:
        """The parts of a dotted name are not references of their own."""
        while isinstance(node, cst.Attribute):
            self.skip.add(id(node))
            node = node.value
        self.skip.add(id(node))

    def _shadow(self, names: list[str]) -> None:
        for name in names:
            self.scopes[-1][name] = UNKNOWN

    # ------------------------------------------------------------- scopes

    def visit_Annotation(self, node) -> None:
        self.annotation += 1

    def leave_Annotation(self, node) -> None:
        self.annotation -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._shadow([node.name.value])         # a local def hides an import
        self.scopes.append({})
        self.kinds.append("function")
        self._shadow(_param_names(node.params))

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.scopes.pop()
        self.kinds.pop()

    def visit_Lambda(self, node: cst.Lambda) -> None:
        self.scopes.append({})
        self.kinds.append("function")
        self._shadow(_param_names(node.params))

    def leave_Lambda(self, node: cst.Lambda) -> None:
        self.scopes.pop()
        self.kinds.pop()

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._shadow([node.name.value])
        bases: list[str] = []
        for b in node.bases:
            self.base_nodes.add(id(b.value))
            self._mark_chain(b.value)
            bases += self._names(b.value)
        self.class_stack.append((node.name.value, bases))
        pre = self.pre.classes.get(id(node), {})
        self.class_pre.append(pre)
        self.scopes.append(dict(pre))
        self.kinds.append("class")
        for q in bases:
            ids = self.index.constructor(q) | self.index.direct(q)
            self._add_reference(node, q, ids, "subclass")

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.class_stack.pop()
        self.class_pre.pop()
        self.scopes.pop()
        self.kinds.pop()

    def visit_Assign(self, node: cst.Assign) -> None:
        for t in node.targets:
            self._assign(t.target, node.value)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if node.value is not None:
            self._assign(node.target, node.value)

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        self._shadow(_target_names(node.target))

    def visit_NamedExpr(self, node: cst.NamedExpr) -> None:
        self._assign(node.target, node.value)

    def visit_With(self, node: cst.With) -> None:
        for item in node.items:
            if item.asname is not None:
                self._assign(item.asname.name, item.item)

    def visit_For(self, node: cst.For) -> None:
        self._shadow(_target_names(node.target))

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
        if node.name is not None:
            self._shadow(_target_names(node.name.name))

    def visit_ListComp(self, node) -> None:
        self._enter_comprehension(node.for_in)

    def visit_SetComp(self, node) -> None:
        self._enter_comprehension(node.for_in)

    def visit_DictComp(self, node) -> None:
        self._enter_comprehension(node.for_in)

    def visit_GeneratorExp(self, node) -> None:
        self._enter_comprehension(node.for_in)

    def leave_ListComp(self, node) -> None:
        self._leave_scope()

    def leave_SetComp(self, node) -> None:
        self._leave_scope()

    def leave_DictComp(self, node) -> None:
        self._leave_scope()

    def leave_GeneratorExp(self, node) -> None:
        self._leave_scope()

    def _enter_comprehension(self, for_in: cst.CompFor) -> None:
        self.scopes.append({})
        self.kinds.append("function")
        clause: cst.CompFor | None = for_in
        while clause is not None:
            self._shadow(_target_names(clause.target))
            clause = clause.inner_for_in

    def _leave_scope(self) -> None:
        self.scopes.pop()
        self.kinds.pop()

    def _assign(self, target, value) -> None:
        """A name bound in the current scope; `self.x` bindings are the
        prescan's, for the whole class, whatever the order of methods. A
        class attribute (`client = Client()` in the class body) is also
        what `self.client` names in the methods, unless a method assigns
        `self.client` itself, which then governs."""
        if isinstance(target, cst.Name) and self.kinds[-1] == "class":
            self._bind(self.scopes[-1], target.value, value)
            key = f"self.{target.value}"
            if key not in self.class_pre[-1]:
                self.scopes[-1][key] = self.scopes[-1][target.value]
            return
        if (isinstance(target, cst.Attribute)
                and isinstance(target.value, cst.Name)
                and target.value.value == "self"):
            return                          # the prescan's, for the class
        self._bind_target(self.scopes[-1], target, value)

    # ------------------------------------------------------------ imports

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            mod = _dotted(alias.name)
            if mod is None or not self.index.ours(mod):
                continue
            self.imports_package = True
            ids, _ = self.index.settled(self.index.direct(mod), IMPORT_KINDS)
            self._add(alias, mod, ids, RESOLVED, "import")
            ids, prefix = self.index.removed_prefix(mod)
            self._add(alias, prefix, ids, RESOLVED, "import")

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.relative or node.module is None:
            return
        mod = _dotted(node.module)
        if mod is None or not self.index.ours(mod):
            return
        self.imports_package = True
        ids, _ = self.index.settled(self.index.direct(mod), IMPORT_KINDS)
        self._add(node, mod, ids, RESOLVED, "import")
        ids, prefix = self.index.removed_prefix(mod)
        self._add(node, prefix, ids, RESOLVED, "import")
        if isinstance(node.names, cst.ImportStar):
            self.star_imports.append((node, mod))
            return
        for alias in node.names:
            name = _dotted(alias.name)
            if name is None:
                continue
            full = f"{mod}.{name}"
            ids, _ = self.index.settled(self.index.direct(full),
                                        IMPORT_KINDS)
            self._add(alias, full, ids, RESOLVED, "import")

    # -------------------------------------------------------------- calls

    def visit_Call(self, node: cst.Call) -> None:
        self._mark_chain(node.func)
        for a in node.args:
            self.arg_nodes.add(id(a.value))
            if isinstance(a.value, (cst.Tuple, cst.List)):
                for el in a.value.elements:
                    self.arg_nodes.add(id(el.value))
        shape = _call_shape(node)
        func = node.func

        # super().__init__(...) or super().method(...) in a project class
        if (isinstance(func, cst.Attribute)
                and isinstance(func.value, cst.Call)
                and isinstance(func.value.func, cst.Name)
                and func.value.func.value == "super" and self.class_stack):
            member = func.attr.value
            for base in self.class_stack[-1][1]:
                if member == "__init__":
                    ids = self.index.constructor(base)
                    owner = self.index.canonical(base)
                else:
                    ids, owner = self.index.method(base, member)
                self._add(node, f"{base}.{member}", ids, REFERENCED, "super",
                          receiver=owner, **shape)
            return

        if any([self._call_by_name(node, q, shape)
                for q in self._names(func)]):
            return
        if not isinstance(func, cst.Attribute):
            return

        # a method on an instance
        member = func.attr.value
        receiver = self._receiver(func.value)
        if receiver == FOREIGN:
            return
        path = f"?.{member}"
        if receiver is not None:
            ids, owner = self.index.method(receiver, member)
            if ids:
                self._add(node, f"{receiver}.{member}", ids, RESOLVED,
                          "method", receiver=owner, **shape)
                return
            if self.index.api is not None:
                return           # the class and its bases were read
            path = f"{receiver}.{member}"
        else:
            names = self._names(func.value)
            if self._foreign(names):
                return           # another package's object
            if names:
                path = f"{names[0]}.{member}"
        self._add(node, path, self.index.leaf(member), UNVERIFIED, "method",
                  **shape)

    def _call_by_name(self, node: cst.Call, q: str, shape: dict) -> bool:
        """A call whose callee libcst resolved to the imported path `q`.
        True when the question is settled, hit or no hit."""
        func = node.func
        ids, prefix = self.index.removed_prefix(q)
        self._add(node, prefix, ids, RESOLVED, "attribute")
        if self.index.is_class(q):
            ids = self.index.constructor(q) | self.index.direct(q)
            self._add(node, q, ids, RESOLVED, "call",
                      receiver=self.index.canonical(q), **shape)
            return True
        cls, _, member = q.rpartition(".")
        if cls and self.index.is_class(cls):
            # Class.method(obj, ...) with self explicit, or Class().method()
            instance = (isinstance(func, cst.Attribute)
                        and isinstance(func.value, cst.Call))
            ids, owner = self.index.method(cls, member)
            if ids:
                explicit = (not instance
                            and self.index.instance_method(q))
                self._add(node, q, ids, RESOLVED, "method", receiver=owner,
                          explicit_self=explicit, **shape)
                return True
            if self.index.api is not None:
                return True      # the class and its bases were read
            self._add(node, q, self.index.leaf(member), UNVERIFIED, "method",
                      **shape)
            return True
        ids = self.index.direct(q)
        if ids:
            self._add(node, q, ids, RESOLVED, "call", **shape)
            return True
        if not self.index.ours(q) or self.index.api is None:
            return False
        # Ours, read, and not changed: settled, unless this is a method on
        # the result of a call (`pkg.connect(...).send()`), whose class the
        # tool does not know; that falls through to unverified.
        return not (isinstance(func, cst.Attribute)
                    and isinstance(func.value, cst.Call))

    # ------------------------------------------------- other references

    def visit_Attribute(self, node: cst.Attribute) -> None:
        if id(node) not in self.skip and self._names(node):
            self._mark_chain(node.value)     # its parts are not references
        self._reference(node)

    def visit_Name(self, node: cst.Name) -> None:
        self._reference(node)

    def _reference(self, node) -> None:
        if (id(node) in self.skip or id(node) in self.base_nodes
                or self.annotation):
            return
        use = "argument" if id(node) in self.arg_nodes else "attribute"
        for q in self._names(node):
            ids, prefix = self.index.removed_prefix(q)
            self._add(node, prefix, ids, RESOLVED, "attribute")
            ids = self.index.direct(q) | self.index.constructor(q)
            if ids:
                self._add_reference(node, q, ids, use)


def python_files(project: pathlib.Path
                 ) -> tuple[list[pathlib.Path], list[str]]:
    """Every `.py` under the project, and the directories not entered
    that are worth a word (not caches and `.git`)."""
    files: list[pathlib.Path] = []
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(project):
        here = pathlib.Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            at_top = here == project
            if (d in SKIP_ANYWHERE or (at_top and d in SKIP_AT_TOP)
                    or (here / d / "pyvenv.cfg").exists()):
                if d not in SKIP_SILENT:
                    skipped.append(
                        (here / d).relative_to(project).as_posix())
            else:
                keep.append(d)
        dirnames[:] = keep
        files += [here / f for f in sorted(filenames) if f.endswith(".py")]
    return sorted(files), skipped


def project_roots(project: pathlib.Path,
                  files: list[pathlib.Path]) -> set[str]:
    """The names an absolute import of the project's own code can start
    with: every directory holding a `.py` file (a package with or without
    an `__init__.py`, wherever it sits: `myapp/`, `src/myapp/`,
    `lib/myapp/`), and the modules directly under the project or its
    `src/`. A name shared with a real package errs towards unknown, never
    towards resolved."""
    out: set[str] = set()
    for f in files:
        if f.parent != project:
            out.add(f.parent.name)
        if f.parent in (project, project / "src"):
            out.add(f.stem)
    return out


def locate_source(text: str, file: str, index: Index
                  ) -> tuple[list[Hit], list[Note], bool]:
    """One file's hits, notes and whether it imports the package; raises
    on a file libcst cannot read."""
    wrapper = MetadataWrapper(cst.parse_module(text), unsafe_skip_copy=True)
    pre = _Prescan(index)
    wrapper.visit(pre)
    finder = _Finder(index, pre)
    wrapper.visit(finder)
    pending = [p for p in finder.pending
               if p[3] != UNVERIFIED or finder.imports_package]
    if not pending and not finder.star_imports:
        return [], [], finder.imports_package
    positions = wrapper.resolve(PositionProvider)
    lines = text.splitlines()

    def source(line: int) -> str:
        return lines[line - 1].strip() if 0 < line <= len(lines) else ""

    hits: dict[tuple, Hit] = {}
    for node, path, ids, confidence, use, extra in pending:
        start = positions[node].start
        column = start.column + 1          # 1-based, as editors show it
        key = (start.line, column, use, confidence)
        old = hits.get(key)
        if old is not None:
            ids = set(old.change_ids) | ids
            extra = {k: getattr(old, k) for k in
                     ("receiver", "explicit_self", "positional", "keywords",
                      "star_args")}
            path = old.path
        hits[key] = Hit(file, start.line, column, source(start.line), path,
                        tuple(sorted(ids)), confidence, use, **extra)
    notes = [Note(file, f"line {positions[node].start.line}: `from {mod} "
                  "import *`: the tool cannot see which names this file "
                  "uses through it") for node, mod in finder.star_imports]
    return [hits[k] for k in sorted(hits)], notes, finder.imports_package


def locate(project: str | pathlib.Path, changes: list[Change],
           api: OldApi | None = None) -> Located:
    """Every hit in the project for the given changes."""
    root = pathlib.Path(project).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    files, skipped = python_files(root)
    own = project_roots(root, files)
    index = Index(changes, api, own)
    roots = tuple(sorted(index.roots))
    mentions = re.compile(r"\b(%s)\b" % "|".join(re.escape(r) for r in roots))
    names = "|".join(re.escape(r) for r in sorted(own)) or "(?!)"
    via_project = re.compile(
        r"^\s*(?:from\s+(?:\.|(?:%s)\b)|import\s+(?:%s)\b)" % (names, names),
        re.M)
    hits: list[Hit] = []
    notes: list[Note] = []
    parsed = via = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            notes.append(Note(rel, "not UTF-8; not read"))
            continue
        if not roots or not mentions.search(text):
            if via_project.search(text):
                via += 1
            continue
        parsed += 1
        try:
            file_hits, file_notes, imports = locate_source(text, rel, index)
        except cst.ParserSyntaxError as e:
            first = str(e).splitlines()[0] if str(e) else "syntax error"
            notes.append(Note(rel, f"not parsed: {first}"))
            continue
        except Exception as e:  # noqa: BLE001 - one file never stops the run
            first = (str(e).splitlines() or [""])[0][:120]
            notes.append(Note(rel, f"not analysed: {type(e).__name__}: "
                                   f"{first}".rstrip(": ")))
            continue
        hits += file_hits
        notes += file_notes
        if not imports and via_project.search(text):
            via += 1
    return Located(hits, notes, len(files), parsed, via, roots,
                   tuple(skipped))
