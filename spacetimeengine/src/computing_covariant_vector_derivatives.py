# Creating symbolic covariant derivative computation in polar coordinates using SymPy
import sympy as sp
from pathlib import Path
import inspect

# Optional: show source file for the top-level diff wrapper
print("sympy.diff source file:", sp.diff.__code__.co_filename)
print("sympy.diff first lines:\n", "\n".join(inspect.getsource(sp.diff).splitlines()[:15]))

# Step 1: Define polar coordinates
r, theta = sp.symbols("r theta")
coords = [r, theta]

# Step 2: Metric tensor in polar coordinates: diag(1, r^2)
g = sp.Matrix([[1, 0], [0, r**2]])

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

# Step 3: Christoffel symbols Γ^k_{ij}
Gamma = christoffel_2nd(g, coords)

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
