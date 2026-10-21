"""
A pretend installed package, version 1.0.0, for scripts/test_fixer_locate.py
(NOTES §38.3). It is never imported: the fixer reads it with griffe, as it
reads the old version of a real package, to learn which class defines a
method and what a function's parameters are. The names are re-exported
here the way redis exports `redis.Redis`, so that the short path a project
writes (`alphalib.Client`) differs from the defining one
(`alphalib.core.Client`).
"""

from . import compat
from .core import Base, Child, Client, connect, helper
from .util import LIMIT, RATE, Pool, ping

__all__ = ["Base", "Child", "Client", "compat", "connect", "helper", "LIMIT",
           "RATE", "Pool", "ping"]
