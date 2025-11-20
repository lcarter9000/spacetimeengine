import sympy as sp, inspect
from sympy.solvers.ode import dsolve, classify_ode
from sympy import Eq, Function, symbols
x = symbols('x')
y = Function('y')
ode = Eq(sp.diff(y(x), x), 3*y(x))
print(classify_ode(ode, y(x)))
print(inspect.getsource(dsolve).splitlines()[:40])