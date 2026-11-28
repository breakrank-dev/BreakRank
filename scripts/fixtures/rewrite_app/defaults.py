"""R2's cases (NOTES §38.15), one call per line against alphalib.opts: the
expected pin of each, or the reason it is withheld, is written in
scripts/test_fixer_rewrite.py before the code ran."""
import alphalib.opts
from alphalib.opts import Session, emit, fetch, shape, ship, wrap

a = fetch("u")
b = fetch("u", 5, 9)
c = fetch("u", retries=1, verify=False, pool=None, timeout=1, tags=(1,))
d = fetch("u", **kw)
e = fetch(*args)
f = shape(1)
g = shape(1, 2, 3)
h = Session("h")
i = Session("h", 1, protocol=3)
j = h.get("k")
k = Session.get(h, "k")
m = ship("p")
n = alphalib.opts.fetch("u", timeout=2)
o = fetch("u", 1, 2, 3, 4, 5, 6, 7, 8, 9)
p = fetch(
    "u",
    retries=1,
)
q = fetch(
    "u",  # the url
    retries=1,
)
r = fetch(
    "u",
    retries=1,  # retry once
)
s = wrap(1)
t = emit(x for x in y)
u = fetch()
v = fetch(
    "u",

    retries=1,
)
