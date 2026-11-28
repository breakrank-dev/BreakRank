"""R2's functions (NOTES §38.15): defaults that 2.0.0 changes, in the
shapes the data has. A `None` that becomes a named constant (redis's
socket timeouts), a number that changes, a boolean, a string that changes
only in spelling, an old default that is a name and not a literal, a
mutable empty container, a tuple, a positional-only parameter with a
default, a keyword-only one, a parameter made required, a function that
also gains a required parameter, a method whose default changes, a
function whose first parameter moves (for a generator argument), and a
positional-only parameter made required, which no pin can answer."""

DEFAULT_LIMIT = 10
SENTINEL = object()


def fetch(url, timeout=None, retries=3, verify=True, mode='fast',
          limit=DEFAULT_LIMIT, hooks=[], pool=None, tags=()):
    return (url, timeout, retries, verify, mode, limit, hooks, pool, tags)


def shape(a, b=1, /, c=2, *, d=1e3):
    return (a, b, c, d)


def ship(path, compress=False):
    return (path, compress)


def emit(items, sep=","):
    return sep.join(items)


def wrap(x, y=2, /, z=3):
    return (x, y, z)


class Session:
    def __init__(self, host, port=6379, *, protocol=2, keepalive=None):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.keepalive = keepalive

    def get(self, key, default=None):
        return default
