"""
Track B — which symbols does the ecosystem actually use?

This is the heart of the project, because it is where the labels come from.
Track A produced 15,000 candidate changes with no way to tell which matter.
This file reads the SOURCE CODE of downstream packages and records which
symbols they touch. A change to a symbol nobody imports cannot break anyone;
a change to a symbol 40 packages import probably breaks someone. That is the
label, and no human ever writes it.

We never run anybody's code. Python's `ast` module parses source into a tree,
and we walk the tree. Three node types tell the whole story:

    import numpy as np          Import       -> remember np means numpy
    from click import echo      ImportFrom   -> click.echo is USED, right here
    np.linalg.solve(...)        Attribute    -> numpy.linalg.solve is USED

Note that a `from x import y` counts as usage at the import line itself.
That is deliberate, not lazy: if `y` is removed, the import raises
ImportError before a single line of the importer runs. Importing a symbol
IS depending on it.

Try it on one package:

    python ml/ingest/usage_index.py requests
"""

import ast
import pathlib


class UsageVisitor(ast.NodeVisitor):
    """Find which tracked external symbols ONE Python file touches."""

    def __init__(self, tracked: set[str]):
        self.tracked = tracked            # module roots we care about: {"click", "numpy", ...}
        self.aliases: dict[str, str] = {} # local name -> real dotted path
        self.used: set[str] = set()

    # import numpy            -> aliases["numpy"] = "numpy",  used: numpy
    # import numpy as np      -> aliases["np"]    = "numpy",  used: numpy
    # import pandas.testing   -> aliases["pandas"]= "pandas", used: pandas.testing
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            if root in self.tracked:
                # `import a.b` binds the name `a` locally; `import a.b as c`
                # binds `c` to the full path. Getting this wrong misroutes
                # every attribute chain that follows.
                if a.asname:
                    self.aliases[a.asname] = a.name
                else:
                    self.aliases[root] = root
                self.used.add(a.name)
        self.generic_visit(node)

    # from requests import Session        -> used: requests.Session
    # from click.utils import LazyFile    -> used: click.utils.LazyFile
    # from numpy import array as arr      -> aliases["arr"] = "numpy.array"
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module or node.level:      # `from . import x` is internal
            self.generic_visit(node)
            return
        root = node.module.split(".")[0]
        if root in self.tracked:
            for a in node.names:
                if a.name == "*":
                    self.used.add(node.module)  # best we can honestly say
                    continue
                full = f"{node.module}.{a.name}"
                self.aliases[a.asname or a.name] = full
                self.used.add(full)
        self.generic_visit(node)

    # np.linalg.solve  ->  numpy.linalg.solve
    # Walks from the outermost attribute inward until it hits a plain name,
    # then substitutes what that name really means.
    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts: list[str] = []
        cur: ast.expr = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            base = self.aliases.get(cur.id)
            if base:
                self.used.add(base + "." + ".".join(reversed(parts)))
        self.generic_visit(node)


def scan_package(root: pathlib.Path, tracked: set[str]) -> set[str]:
    """
    Every tracked symbol this package's source touches, as ONE set.

    A set, not a list — within one package we do not care how many times.
    Files that will not parse (Python 2 relics, templates, fixtures full of
    deliberately broken syntax) are skipped; they cannot import anything.
    """
    found: set[str] = set()
    for py in root.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError, RecursionError):
            continue
        v = UsageVisitor(tracked)
        v.visit(tree)
        found |= v.used
    return found


if __name__ == "__main__":
    import sys

    from download import download_and_extract, list_releases

    package = sys.argv[1] if len(sys.argv) > 1 else "requests"
    tracked = {"urllib3", "idna", "certifi", "charset_normalizer", "click",
               "numpy", "pandas", "requests"}

    rel = list_releases(package, last_n=1)[-1]
    path = download_and_extract(rel["url"], pathlib.Path(f"data/usage_tmp/{package}"))
    used = scan_package(path, tracked)

    print(f"\n{package} {rel['version']} touches {len(used)} tracked symbols:\n")
    for s in sorted(used)[:40]:
        print("  ", s)
