import os
from sympy import symbols, Function, simplify, Matrix, diag, sin, cos, diff
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensorhead, TensExpr
from sympy.printing.latex import latex
from sympy.abc import r, theta, phi, t
from fpdf import FPDF

# Create output directory if it doesn't exist
os.makedirs("/mnt/data", exist_ok=True)

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

# Helper function to add section
def add_section(title, content):
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt=title, ln=True)
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)

# Section 1: Introduction
intro = """
This worksheet walks through the Einstein field equations using the Schwarzschild metric as an example.
We derive the key tensors involved in general relativity using symbolic index notation and Python/SymPy code snippets.
"""
add_section("Introduction", intro)

# Section 2: Schwarzschild Metric
metric_expr = """
The Schwarzschild metric in natural units (G = c = 1) is given by:

ds² = -(1 - 2M/r) dt² + (1 - 2M/r)^(-1) dr² + r² dθ² + r² sin²θ dφ²

In matrix form:

g_{μν} = diag(-(1 - 2M/r), (1 - 2M/r)^(-1), r², r² sin²θ)
"""
add_section("Schwarzschild Metric", metric_expr)

# Section 3: Metric Tensor
metric_code = """
from sympy import symbols, diag, sin
M, r, theta = symbols('M r theta')
g = diag(-(1 - 2*M/r), 1/(1 - 2*M/r), r**2, r**2*sin(theta)**2)
g
"""
add_section("Metric Tensor g_{μν}", "Metric tensor components in Schwarzschild coordinates:\n" + metric_code)

# Section 4: Christoffel Symbols
christoffel_expr = """
The Christoffel symbols are defined as:
Γ^λ_{μν} = 1/2 g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} - ∂_σ g_{μν})

They represent the connection coefficients for parallel transport.
"""
add_section("Christoffel Symbols Γ^λ_{μν}", christoffel_expr)

# Section 5: Riemann Tensor
riemann_expr = """
The Riemann curvature tensor is defined as:
R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} - Γ^ρ_{νλ} Γ^λ_{μσ}

It encodes the intrinsic curvature of spacetime.
"""
add_section("Riemann Tensor R^ρ_{σμν}", riemann_expr)

# Section 6: Ricci Tensor
ricci_expr = """
The Ricci tensor is obtained by contracting the Riemann tensor:
R_{μν} = R^λ_{μλν}
"""
add_section("Ricci Tensor R_{μν}", ricci_expr)

# Section 7: Ricci Scalar
ricci_scalar_expr = """
The Ricci scalar is the trace of the Ricci tensor:
R = g^{μν} R_{μν}
"""
add_section("Ricci Scalar R", ricci_scalar_expr)

# Section 8: Einstein Tensor
einstein_tensor_expr = """
The Einstein tensor is defined as:
G_{μν} = R_{μν} - 1/2 g_{μν} R

It appears on the left-hand side of the Einstein field equations:
G_{μν} = 8π T_{μν}
"""
add_section("Einstein Tensor G_{μν}", einstein_tensor_expr)

# Save PDF
pdf.output("/mnt/data/einstein_field_equations_worksheet.pdf")
print("Worksheet generated successfully.")