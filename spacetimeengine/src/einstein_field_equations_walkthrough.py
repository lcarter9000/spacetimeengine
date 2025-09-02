import os
from sympy import symbols, Function, simplify, Matrix, diag, sin, cos, diff
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensorhead, TensExpr
from sympy.printing.latex import latex
from sympy.abc import r, theta, phi, t
from fpdf import FPDF

# Create output directory if it doesn't exist
os.makedirs("C:/Users/lcart/Documents/GitHub/spacetime-toolkit/mnt/data", exist_ok=True)

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

ds^2 = -(1 - 2M/r) dt^2 + (1 - 2M/r)^(-1) dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2

In matrix form:

g_{mu nu} = diag(-(1 - 2M/r), (1 - 2M/r)^(-1), r^2, r^2 sin^2(theta))
"""
add_section("Schwarzschild Metric", metric_expr)

# Section 3: Metric Tensor
metric_code = """
from sympy import symbols, diag, sin
M, r, theta = symbols('M r theta')
g = diag(-(1 - 2*M/r), 1/(1 - 2*M/r), r**2, r**2*sin(theta)**2)
g
"""
add_section("Metric Tensor g_{mu nu}", "Metric tensor components in Schwarzschild coordinates:\n" + metric_code)

# Section 4: Christoffel Symbols
christoffel_expr = """
The Christoffel symbols are defined as:
Gamma^lambda_{mu nu} = 1/2 g^{lambda sigma} (partial_mu g_{nu sigma} + partial_nu g_{mu sigma} - partial_sigma g_{mu nu})

They represent the connection coefficients for parallel transport.
"""
add_section("Christoffel Symbols Gamma^lambda_{mu nu}", christoffel_expr)

# Section 5: Riemann Tensor
riemann_expr = """
The Riemann curvature tensor is defined as:
R^rho_{sigma mu nu} = partial_mu Gamma^rho_{nu sigma} - partial_nu Gamma^rho_{mu sigma} + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}

It encodes the intrinsic curvature of spacetime.
"""
add_section("Riemann Tensor R^rho_{sigma mu nu}", riemann_expr)

# Section 6: Ricci Tensor
ricci_expr = """
The Ricci tensor is obtained by contracting the Riemann tensor:
R_{mu nu} = R^lambda_{mu lambda nu}
"""
add_section("Ricci Tensor R_{mu nu}", ricci_expr)

# Section 7: Ricci Scalar
ricci_scalar_expr = """
The Ricci scalar is the trace of the Ricci tensor:
R = g^{mu nu} R_{mu nu}
"""
add_section("Ricci Scalar R", ricci_scalar_expr)

# Section 8: Einstein Tensor
einstein_tensor_expr = """
The Einstein tensor is defined as:
G_{mu nu} = R_{mu nu} - 1/2 g_{mu nu} R

It appears on the left-hand side of the Einstein field equations:
G_{mu nu} = 8pi T_{mu nu}
"""
add_section("Einstein Tensor G_{mu nu}", einstein_tensor_expr)

# Save PDF
pdf.output("C:/Users/lcart/Documents/GitHub/spacetime-toolkit/mnt/data/einstein_field_equations_worksheet.pdf")
print("Worksheet generated successfully.")