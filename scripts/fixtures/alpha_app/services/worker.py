"""Names reached through the defining module and through the class."""
import alphalib.core
from alphalib.core import Base, Client as C

import requests


def build():
    c = C("h", port=1)
    Base.send(c, b"p", 3)
    alphalib.core.Client.send(c, b"p")
    s = requests.Session()
    s.send(b"not ours")
    return c


def later():
    return worker_client.send(b"late")


worker_client = C("later")
