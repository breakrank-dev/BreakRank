"""The pretend package's clients. `send` lives on the base class, as
redis's `set` lives on a commands mixin, so a change recorded on
`alphalib.core.Base.send` must match a call on a `Client` instance."""


class Base:
    def send(self, payload, timeout=10):
        return payload

    def close(self):
        return None


class Client(Base):
    def __init__(self, host, port=6379, db=0, retries=3):
        self.host = host
        self.port = port
        self.db = db
        self.retries = retries

    def get(self, key, default=None):
        return default

    @classmethod
    def from_url(cls, url, retries=3):
        return cls(url, retries=retries)


class Child(Client):
    """Inherits `__init__`: its parameters are Client's."""


def connect(host, port=1, timeout=5):
    return Client(host, port)


def helper(a, b, /, c, *, d=1):
    return (a, b, c, d)


def _internal():
    return None
