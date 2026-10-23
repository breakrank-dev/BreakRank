"""R1's cases (NOTES §38.4), one call per line: the expected rewrite of
each, or the reason it is withheld, is written in
scripts/test_fixer_rewrite.py before the code ran."""
import alphalib
from alphalib import Base, Client, connect, helper
from alphalib.core import Child
from alphalib.util import spread

a = Client("localhost", 6380, 2)
b = Client("localhost", 6380)
c = Client("localhost", port=6380, db=2)
d = Client("localhost")
e = Client("h", 1, 2, 3)
f = Client(*d.args)
g = Client("h", 1, db=2)
h = Client("h", 1, 2, 3, 4)
i = a.send(b"x", 5)
j = Base.send(a, b"x", 5)
k = a.send(b"x")
m = helper(1, 2, 3)
n = connect("h", 2, timeout=9)
o = Child("h", 1, 2)
p = Client.from_url("u", 5)
q = alphalib.Client("h", 1)
r = Client("h", 1,  # the port
           2)
s = spread("a", "b", "c", True)
t = Client.parse("u", True)
u = Client("h", 1, port=2)
v = Client("h", 1).send(b"t", 5)
w = Client("h", 1, 2, 3, retries=9)
x = helper(1, 2, 3, 4)
y = Client("h", 1)("x")
z = Client("h", 1)("x").send(b"t", 5)
r = Client("h", 1)("x")
r.send(b"t", 6)
