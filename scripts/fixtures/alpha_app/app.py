"""The fixture project for scripts/test_fixer_locate.py: every way a
project names alphalib's changed symbols, one per line, each line's
expected hit written in the test before the tool ran (NOTES §38.3)."""
from functools import partial

import alphalib
import alphalib.util as util
from alphalib import Client, connect
from alphalib.core import _internal
from alphalib.util import LIMIT as CAP

client = Client("localhost", 6380, 2)
client.send(b"x", 5)
other = connect("h", 2, timeout=9)
cap = CAP + util.LIMIT
pool = alphalib.Pool()
pool.acquire()
fn = partial(Client, "h")
alphalib.util.ping("h", 2)
print(_internal)


def make(host):
    c = Client(host)
    c.get("k")
    return c.send(payload=b"y")


def unknown(x):
    x.send(b"z")
    x.acquire()
    isinstance(x, Client)
    return alphalib.helper(1, 2, 3)


def temp(name: Client):
    return Client("h").send(b"t")


class MyClient(Client):
    def __init__(self, host):
        super().__init__(host, 1)
        self.pool = alphalib.Pool()

    def run(self):
        self.pool.acquire(True)
        return self.send(b"q", 1)


with Client("w") as cw:
    cw.send(b"w", timeout=1)
