# Creating symbolic covariant derivative computation in polar coordinates using SymPy
import sympy as sp
from pathlib import Path

print(sp.diff.__code__.co_filename)  # Debug: Check SymPy installation path

"""Error correction: The import path is not  available in your SymPy version. 
   Remove that import and compute Christoffel symbols directly. Also fix the 
   Linux-only output path.
"""

# Step 1: Define polar coordinates
r, theta = sp.symbols("r theta")
coords = [r, theta]

# Step 2: Construct the metric tensor g_ij for polar coordinates
# In polar coordinates: ds^2 = dr^2 + r^2 dθ^2
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
                # Debug: print only nonzero components
                if s != 0:
                    print(f"Gamma^{k}_[{i}{j}] = {s}")
                Gamma[k][i, j] = s
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

""" Layex format  of the output:
   \nabla_{r} v_{r} = 4
   \nabla_{theta} v_{r} = - \frac{20 \theta}{r}
   \nabla_{r} v_{theta} = - \frac{20 \theta}{r}
   \nabla_{theta} v_{theta} = 4 r^{2} + 20
"""


"""Non-LaTeX version of the output expressions, written in ordinary math notation:
- Partial derivative of v_r with respect to r:
  ∂v_r / ∂r = 4
- Partial derivative of v_r with respect to \theta :
  ∂v_r / ∂θ = - (20·θ) / r
- Partial derivative of v_{\theta } with respect to r:
  ∂v_θ / ∂r = - (20·θ) / r
- Partial derivative of v_{\theta } with respect to \theta :
  ∂v_θ / ∂θ = 4·r² + 20
"""
