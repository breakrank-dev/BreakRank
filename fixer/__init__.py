"""
BreakRank's fixer (NOTES §38): for one upgrade of one package, list the
lines of a project that each changed name touches, in the site's order,
rewrite the ones whose fix is mechanical, and print a diff to review.

    python -m fixer <package> --from <version> --to <version> [project]

It runs offline, from a bundle of every analysed change built from the
pipeline's own files by the loader's own rules (fixer/bundle.py,
scripts/build_bundle.py); it never edits without printing the diff, never
runs the project's code or the package's unless told to run the tests,
and never guesses a replacement. Week 2 of the plan is the bundle and
locate; the rules R1 and R2 follow (§38.4).
"""

__version__ = "0.1.0"
