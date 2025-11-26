# Creating symbolic covariant derivative computation in polar coordinates using SymPy
import sympy as sp
from pathlib import Path

# Step 1: Define polar coordinates
r, theta = sp.symbols("r theta")
coords = [r, theta]

# Step 2: Construct the metric tensor g_ij for polar coordinates
# In polar coordinates: ds^2 = dr^2 + r^2 dθ^2 (Pythagorean theorem in curved coordinates)
g = sp.Matrix([[1, 0], [0, r**2]])

def christoffel_2nd(g: sp.Matrix, coords): # Compute Christoffel symbols of the second kind
    n = len(coords)
    g_inv = g.inv()
    Gamma = [sp.MutableDenseMatrix.zeros(n) for _ in range(n)]  # Γ^k_{ij} as list over k of (i,j) matrices
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = 0 # Summation over l
                for l in range(n):
                    s += sp.Rational(1, 2) * g_inv[k, l] * (
                        sp.diff(g[l, j], coords[i]) + sp.diff(g[l, i], coords[j]) - sp.diff(g[i, j], coords[l])
                    )
                    print(f"Intermediate Gamma^{k}_[{i}{j}] after l={l}: {s}")
                Gamma[k][i, j] = sp.simplify(s)
    return [sp.Matrix(Gamma[k]) for k in range(n)]

# Step 3: Compute Christoffel symbols of the second kind
Gamma = christoffel_2nd(g, coords)

# Step 4: Define covariant vector field v_i with components v_r = 4r, v_θ = 20θ
v = sp.Matrix([4*r, 20*theta])

# Step 5: Compute covariant derivatives ∇_j v_i
# ∇_j v_i = ∂_j v_i - Γ^k_{ij} v_k
n = len(coords)
nabla_v = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        partial = sp.diff(v[i], coords[j])
        correction = sum(Gamma[k][i, j] * v[k] for k in range(n))
        nabla_v[i][j] = sp.simplify(partial - correction)

# Step 6: Prepare LaTeX output
latex_output = []
for i in range(n):
    for j in range(n):
        expr = sp.simplify(nabla_v[i][j])
        latex_output.append(f"\\nabla_{{{coords[j]}}} v_{{{coords[i]}}} = {sp.latex(expr)}")

# Save LaTeX output to a text file next to this script (Windows-friendly)
output_path = Path(__file__).with_name("covariant_derivatives_latex.txt")
with output_path.open("w", encoding="utf-8") as f:
    for line in latex_output:
        f.write(line + "\n")

print(f"Computed covariant derivatives and saved LaTeX expressions to: {output_path}")