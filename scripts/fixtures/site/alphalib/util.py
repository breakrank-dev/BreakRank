"""The pretend package's constants and pool."""

LIMIT = 10
RATE = 0.5


class Pool:
    def acquire(self, block=True):
        return None


def ping(target, count=1):
    return count


def spread(first, *rest, flag=False):
    return (first, rest, flag)
