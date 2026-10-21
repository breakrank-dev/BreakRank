"""A module the old version has and 2.0.0 removes (the test's change 13),
so every path through it is a hit; it imports another package, so a path
like `alphalib.compat.json.dumps` is one griffe cannot resolve statically,
which must answer None, not stop the run."""
import json


def old_ping(target):
    return json.dumps(target)
