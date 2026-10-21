"""A file the tool cannot see through: a star import of the package."""
from alphalib.util import *  # noqa: F403


def go():
    return ping("h")  # noqa: F405
