"""
The frozen agreement with Varad's API side, in ONE importable place.

His FastAPI service and this pipeline never call each other's code — they
meet only in the Postgres database, so the exact meaning of every shared
column IS the interface. This file holds the definitions both sides froze
on 3 September 2026:

    docs/api-contract.md   (feat/database-schema branch) — the full contract
    pipelinechangesrequired.md                           — the summary Varad sent

Nothing in here may drift quietly. If a definition needs to change, that is
a conversation with Varad and a PR touching docs/api-contract.md — not an
edit here first. "Frozen" means it doesn't change without a conversation,
not that it can't change.

Run this file to check the rules against Varad's own worked examples:

    python ml/contract.py
"""

from packaging.version import InvalidVersion, Version


def is_private_symbol(symbol_path: str, dunder_all: set[str] | None = None) -> bool:
    """Private if any path component starts with a single underscore.

    Dunder names are excluded. Membership in __all__ overrides to public.

    This is Varad's reference implementation, VERBATIM (contract change 1).
    The API hides is_private rows unless ?include_private=true, so if our
    definition drifted from his, ~40% of rows would be silently shown or
    hidden wrongly — and no query would ever error to tell anyone.

        pandas.DataFrame.append          -> False  (no underscore anywhere)
        pandas.core.frame._parse_header  -> True   (leaf is _private)
        pandas.core._internals.Block     -> True   (a MIDDLE part is private:
                                                    nothing behind it is
                                                    reachable from outside)
        numpy.__version__                -> False  (dunder, excluded)
        attrs._make.attrib               -> False  IF 'attrib' is in
                                                   attrs.__all__ — the package
                                                   exported it on purpose

    The __all__ check uses the LEAF name, because the leaf is what a package
    exports. And note the dunder test is startswith("__") only: under the
    frozen rule a name like __mangled (no trailing underscores) counts as
    public. Our Day-3 rule called that private; the shared column follows
    the contract. is_dunder, our internal ML feature, stays ours.
    """
    if dunder_all and symbol_path.split(".")[-1] in dunder_all:
        return False

    for component in symbol_path.split("."):
        if component.startswith("_") and not component.startswith("__"):
            return True
    return False


def bump_type(version_from: str, version_to: str) -> str:
    """'major' | 'minor' | 'patch' | 'other' — what the version number PROMISED.

    This lands in release.bump_type, and it doubles as the semver baseline
    the model has to beat: the promise says patch releases are safe, and
    roughly a third of the time the promise is wrong. That gap is the
    entire project.

    PEP 440 parsing, never string maths — 2.10.0 is newer than 2.9.0.
    Anything that is not a clean step (epochs, post-releases, four-part
    versions, unparseable strings) is 'other', matching the CHECK
    constraint on the release table.
    """
    try:
        old = Version(version_from).release
        new = Version(version_to).release
    except InvalidVersion:
        return "other"
    old3 = (old + (0, 0, 0))[:3]
    new3 = (new + (0, 0, 0))[:3]
    if new3[0] != old3[0]:
        return "major"
    if new3[1] != old3[1]:
        return "minor"
    if new3[2] != old3[2]:
        return "patch"
    return "other"


if __name__ == "__main__":
    # Varad's worked-examples table, as executable checks. If this prints
    # only the final line, the contract holds on both sides of the join.
    cases = [
        ("pandas.DataFrame.append", None, False),
        ("pandas.core.frame._parse_header", None, True),
        ("pandas.core._internals.Block", None, True),
        ("pandas._libs.tslibs.Timestamp", None, True),
        ("numpy.__version__", None, False),
        ("attrs._make.attrib", {"attrib", "define"}, False),
        ("attrs._make.attrib", set(), True),         # no __all__ -> no override
        ("attrs._make._compile", {"attrib"}, True),  # override is leaf-exact
    ]
    for path, dall, want in cases:
        got = is_private_symbol(path, dall)
        assert got is want, f"{path} with __all__={dall}: got {got}, want {want}"

    bumps = [
        ("2.1.0", "3.0.0", "major"),
        ("2.1.0", "2.2.0", "minor"),
        ("2.1.3", "2.1.4", "patch"),
        ("2.9.0", "2.10.0", "minor"),   # the string-sort trap, dodged
        ("2.1", "2.1.1", "patch"),      # short versions pad with zeros
        ("1.2.3", "1.2.3.1", "other"),  # four-part step is not a semver step
        ("garbage", "2.0.0", "other"),
    ]
    for vf, vt, want in bumps:
        got = bump_type(vf, vt)
        assert got == want, f"{vf} -> {vt}: got {got}, want {want}"

    print("contract self-test: every case matches Varad's table")
