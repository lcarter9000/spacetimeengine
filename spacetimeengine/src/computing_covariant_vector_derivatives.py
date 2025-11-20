# Creating symbolic covariant derivative computation in polar coordinates using SymPy
import sympy as sp
from pathlib import Path
import inspect
import sys
import pdb



"""
# ┌────────┬────────────────────────────────────────────────────────────┐
# │ Command│ Description                                                │
# ├────────┼────────────────────────────────────────────────────────────┤
# │ c      │ Continue execution until next breakpoint or trace call     │
# │ n      │ Step to the next line in the current function              │
# │ s      │ Step into the next function call                           │
# │ r      │ Continue until the current function returns                │
# │ q      │ Quit the debugger and stop execution                       │
# │ l      │ List source code around the current line                   │
# │ p var  │ Print the value of variable `var`                          │
# │ b line │ Set breakpoint at line number `line`                       │
# │ b func │ Set breakpoint at function `func`                          │
# │ cl     │ Clear all breakpoints                                      │
# │ h      │ Show help on commands                                      │
# └────────┴────────────────────────────────────────────────────────────┘
"""

def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        print(f"\n🔍 Pausing at function: {func_name} ({filename}:{lineno})")
        pdb.set_trace()  # Pause execution here
    return trace_calls  # Continue tracing deeper calls




# Optional: show source file for the top-level diff wrapper
print("sympy.diff source file:", sp.diff.__code__.co_filename)
print("sympy.diff first lines:\n", "\n".join(inspect.getsource(sp.diff).splitlines()[:15]))

# Step 1: Define polar coordinates
r, theta = sp.symbols("r theta")
coords = [r, theta]

# Step 2: Metric tensor in polar coordinates: diag(1, r^2)
g = sp.Matrix([[1, 0], [0, r**2]])

#sys.settrace(trace_calls) # Start tracing function calls

"""
def christoffel_2nd(g: sp.Matrix, coords):
    n = len(coords)
    g_inv = g.inv()
    Gamma = [sp.MutableDenseMatrix.zeros(n) for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = 0
                for l in range(n):
                    term = sp.Rational(1, 2) * g_inv[k, l] * (
                        sp.diff(g[l, j], coords[i]) + sp.diff(g[l, i], coords[j]) - sp.diff(g[i, j], coords[l])
                    )
                    s += term
                s = sp.simplify(s)
                if s != 0:
                    print(f"Gamma^{k}_[{i}{j}] = {s}")
                Gamma[k][i, j] = s
    return [sp.Matrix(Gamma[k]) for k in range(n)]

"""

def christoffel_2nd_fd(g: sp.Matrix, coords, h_r=1e-6, h_theta=1e-6):
    """
    Finite difference approximation of Christoffel symbols Γ^k_{ij}.
    Central differences:
      ∂_x g_{ab} ≈ (g_{ab}(x+h) - g_{ab}(x-h)) / (2h)
    h_r, h_theta are step sizes for r, theta.
    """
    r_sym, theta_sym = coords
    steps = {r_sym: h_r, theta_sym: h_theta}
    n = len(coords)
    g_inv = g.inv()
    Gamma = [sp.MutableDenseMatrix.zeros(n) for _ in range(n)]

    def fd_derivative(expr, var, h):
        # Central difference: (f(var+h) - f(var-h))/(2h)
        return (expr.subs(var, var + h) - expr.subs(var, var - h)) / (2*h)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = 0
                for l in range(n):
                    # Approximate the three metric derivatives
                    if i == 0:  # derivative wrt r
                        dg_lj_di = fd_derivative(g[l, j], r_sym, steps[r_sym])
                    else:
                        dg_lj_di = fd_derivative(g[l, j], theta_sym, steps[theta_sym])
                    if j == 0:
                        dg_li_dj = fd_derivative(g[l, i], r_sym, steps[r_sym])
                    else:
                        dg_li_dj = fd_derivative(g[l, i], theta_sym, steps[theta_sym])
                    if l == 0:
                        dg_ij_dl = fd_derivative(g[i, j], r_sym, steps[r_sym])
                    else:
                        dg_ij_dl = fd_derivative(g[i, j], theta_sym, steps[theta_sym])

                    term = sp.Rational(1, 2) * g_inv[k, l] * (dg_lj_di + dg_li_dj - dg_ij_dl)
                    s += sp.simplify(term)
                s = sp.simplify(s)
                if s != 0:
                    print(f"[FD] Gamma^{k}_[{i}{j}] = {s}")
                Gamma[k][i, j] = s
    return [sp.Matrix(Gamma[k]) for k in range(n)]

# Step 3: Christoffel symbols Γ^k_{ij} via finite differences
Gamma = christoffel_2nd_fd(g, coords)

# Step 4: Covariant vector components v_r = 4r, v_θ = 20θ
v = sp.Matrix([4*r, 20*theta])

# Step 5: Covariant derivatives ∇_j v_i = ∂_j v_i - Γ^k_{ij} v_k
n = len(coords)
nabla_v = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        partial = sp.diff(v[i], coords[j])
        correction = sum(Gamma[k][i, j] * v[k] for k in range(n))
        nabla_v[i][j] = sp.simplify(partial - correction)

# Step 6: Prepare LaTeX and plain output
latex_output = []
plain_output = []
for i in range(n):
    for j in range(n):
        expr = nabla_v[i][j]
        latex_output.append(f"\\nabla_{{{coords[j]}}} v_{{{coords[i]}}} = {sp.latex(expr)}")
        plain_output.append(f"∇_{coords[j]} v_{coords[i]} = {expr}")

# Save outputs
out_path = Path(__file__).with_name("covariant_derivatives_latex.txt")
with out_path.open("w", encoding="utf-8") as f:
    f.write("# Covariant derivatives (LaTeX):\n")
    for line in latex_output:
        f.write(line + "\n")
    f.write("\n# Covariant derivatives (plain):\n")
    for line in plain_output:
        f.write(line + "\n")

print(f"Saved results to {out_path}")

sys.settrace(None)  # Stop tracing after main
