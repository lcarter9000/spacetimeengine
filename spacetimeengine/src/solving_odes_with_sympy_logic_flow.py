import sympy as sp

# Define the variable and function
x = sp.Symbol('x')
y = sp.Function('y')

# Define the differential equation: dy/dx = y
ode = sp.Eq(sp.diff(y(x), x), y(x))

# Solve the ODE
solution = sp.dsolve(ode, y(x))

print("General solution:", solution)

# Apply initial condition y(0) = 1
ics = {y(0): 1}
solution_with_ic = sp.dsolve(ode, y(x), ics=ics)

print("Solution with initial condition:", solution_with_ic)