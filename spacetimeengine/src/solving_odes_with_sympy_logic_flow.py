import sympy as sp

# Define the variable and function
x = sp.Symbol('x')
y = sp.Function('y')

# Define the differential equation: dy/dx = y
ode = sp.Eq(sp.diff(y(x), x), y(x)) # Define the ODE dy/dx = y

# Solve the ODE
solution = sp.dsolve(ode, y(x), n=2) # Solve the ODE without initial conditions

"""
(function) def dsolve(
    eq: Any,
    func: ... = ...,
    hint: ... = ...,
    simplify: ... = ...,
    ics: ... = ...,
    xi: ... = ...,
    eta: ... = ...,
    x0: ... = ...,
    n: ... = ...,
    **kwargs: Any
    For single ordinary differential equation
    It is classified under this when number of equation in eq is one. Usage

    dsolve(eq, f(x), hint) -> Solve ordinary differential equation eq for function f(x), using method hint.

    Details

    eq can be any supported ordinary differential equation
    == 

    f(x) is a function of one variable whose derivatives in that
    ====

    variable make up the ordinary differential equation eq. In
    many cases it is not necessary to provide this; it will be
    autodetected (and an error raised if it could not be detected).

    hint is the solving method that you want dsolve to use. Use
    ====

    classify_ode(eq, f(x)) to get all of the possible hints for an
    ODE. The default hint, default, will use whatever hint is
    returned first by ~sympy.solvers.ode.classify_ode. See
    Hints below for more options that you can use for hint.

    simplify enables simplification by
    ========

    ~sympy.solvers.ode.ode.odesimp. See its docstring for more
    information. Turn this off, for example, to disable solving of
    solutions for func or simplification of arbitrary constants.
    It will still integrate with this hint. Note that the solution may
    contain more arbitrary constants than the order of the ODE with
    this option enabled.

    xi and eta are the infinitesimal functions of an ordinary
    ==========

    differential equation. They are the infinitesimals of the Lie group
    of point transformations for which the differential equation is
    invariant. The user can specify values for the infinitesimals. If
    nothing is specified, xi and eta are calculated using
    ~sympy.solvers.ode.infinitesimals with the help of various
    heuristics.

    ics is the set of initial/boundary conditions for the differential equation.
    ===

    It should be given in the form of {f(x0): x1, f(x).diff(x).subs(x, x2): x3} and so on. For power series solutions, if no initial
    conditions are specified f(0) is assumed to be C0 and the power
    series solution is calculated about 0.

    x0 is the point about which the power series solution of a differential
    ==

    equation is to be evaluated.

    n gives the exponent of the dependent variable up to which the power series
    =

    solution of a differential equation is to be evaluated.
"""

print("General solution:", solution)

# Apply initial condition y(0) = 1
ics = {y(0): 1}
solution_with_ic = sp.dsolve(ode, y(x), ics=ics) # Solve the ODE with initial conditions

print("Solution with initial condition:", solution_with_ic)