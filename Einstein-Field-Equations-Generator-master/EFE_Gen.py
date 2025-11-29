import sympy as sp
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def get_christoffel_symbols(metric, axes): # Calculate Christoffel symbols from metric tensor
    metric_inv = metric.inv() # Inverse of the metric tensor
    christoffel = np.zeros([4, 4, 4], dtype = type(sp.Symbol(''))) # Initialize Christoffel symbols array

    for i in range(4): # Loop over indices 
        for j in range(4): # # Loop over indices
            for k in range(4): # Loop over indices
                for s in range(4): # Summation index
                    christoffel[i][j][k] += metric_inv[s, i] * (sp.diff(metric[s, j], axes[k]) + sp.diff(metric[s, k], 
                        axes[j]) - sp.diff(metric[j, k], axes[i])) # Compute Christoffel symbol component
                christoffel[i][j][k] = christoffel[i][j][k] / 2 # Finalize Christoffel symbol component
                christoffel[i][j][k] = sp.simplify(christoffel[i][j][k]) # Simplify expression
        return christoffel

def get_reimann_tensor(christoffel_symbols, axes): # Calculate Riemann curvature tensor from Christoffel symbols
    reimann = np.zeros([4, 4, 4, 4], dtype = type(sp.Symbol(''))) # Initialize Riemann tensor array
    for i in range(4): # Loop over indices
        for j in range(4): # Loop over indices
            for k in range(4): # Loop over indices
                for l in range(4): # Loop over indices
                    differential_part = 0 # Initialize differential part
                    coeff_sum_part = 0  # Initialize coefficient sum part
                    differential_part = sp.diff(christoffel_symbols[i][j][l], axes[k]) 
                    - sp.diff(christoffel_symbols[i][k][l], axes[j]) # Compute differential part

                    for p in range(4): # Summation index
                        coeff_sum_part += christoffel_symbols[p][j][l] * christoffel_symbols[i][p][k]
                        - christoffel_symbols[p][k][l] * christoffel_symbols[i][p][j] # Compute coefficient sum part
                    reimann[i][j][k][l] = differential_part + coeff_sum_part # Combine parts to get Riemann tensor component
                    reimann[i][j][k][l] = sp.simplify(reimann[i][j][k][l]) # Simplify expression
    return reimann

def get_ricci_tensor(reimann_tensor): #$ Calculate Ricci curvature tensor from Riemann tensor
    ricci = np.zeros([4, 4], dtype=type(sp.Symbol(''))) # Initialize Ricci tensor array
    for i in range(4): # Loop over indices
        for j in range(4): # Loop over indices
            for k in range(4): # Summation index
                ricci[i][j] += reimann_tensor[k, i, k, j] # Compute Ricci tensor component
            ricci[i][j] = sp.simplify(ricci[i][j]) # Simplify expression
    print('\nRicci Curvature Tensor: \n', sp.Matrix(ricci))
    return sp.Matrix(ricci)

def raise_one_index(tensor, metric): # Raise one index of a tensor using the metric tensor
    shape = tensor.shape # Get shape of the tensor
    rank = len(shape) # Get rank of the tensor
    metric_inv = metric.inv() # Inverse of the metric tensor
    raised_tensor = np.zeros(shape, dtype=type(sp.Symbol(''))) # Initialize raised tensor array
    for i in range(4): # Loop over indices
        for j in range(4): # Loop over indices
            for k in range(4): # Summation index
                raised_tensor[i][j] += metric_inv[i, k] * tensor[k, j] # Raise index
            raised_tensor[i][j] = sp.simplify(raised_tensor[i][j]) # Simplify expression
    return sp.Matrix(raised_tensor)

def get_curvature_scalar(raised_ricci): # Calculate curvature scalar from raised Ricci tensor
    curvature_scalar = 0 # Initialize curvature scalar
    for i in range(4): # 
        for j in range(4): # Loop over indices
            curvature_scalar += raised_ricci[i, j] # Compute curvature scalar
    curvature_scalar = sp.simplify(curvature_scalar) # Simplify expression
    print('\nCurvature Scalar: \n', curvature_scalar) # Print curvature scalar
    return curvature_scalar

def conform_compacted_metric(axes):
    return sp.Matrix(([-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sp.sin(axes[1])**2, 0], [0, 0, 0, sp.sin(axes[1])**2 * sp.sin(axes[2])**2]))

def spherical_metric(axes):
    c = sp.Symbol('c')
    return sp.Matrix(([-c**2, 0, 0, 0], [0, 1, 0, 0], [0, 0, axes[1]**2, 0], [0, 0, 0, axes[1]**2 * sp.sin(axes[2]) ** 2]))

def FRW_metric(axes):
    a, k = sp.symbols('a k')
    return sp.Matrix(([-1, 0, 0, 0],[0, a**2/(1-k*axes[1]**2), 0, 0], [0, 0, a**2*axes[1]**2, 0],[0, 0, 0, a**2*axes[1]**2*sp.sin(axes[2])**2]))

def _latex_bmatrix(expr: sp.Matrix) -> str:
    """Convert SymPy Matrix LaTeX to bmatrix for mathtext compatibility."""
    s = sp.latex(expr)
    s = s.replace(r"\left[\begin{matrix}", r"\begin{bmatrix}")
    s = s.replace(r"\end{matrix}\right]", r"\end{bmatrix}")
    return s

def _sanitize_latex(expr) -> str:
    s = sp.latex(expr)
    # remove \left / \right which often break mathtext
    return s.replace(r"\left", "").replace(r"\right", "")

def _matrix_plain(name: str, M: sp.Matrix) -> list[str]:
    rows = []
    for i in range(M.rows):
        row_entries = "  ".join(_sanitize_latex(sp.simplify(M[i, j])) for j in range(M.cols))
        rows.append(f"{name}[{i}, *] = {row_entries}")
    return rows

def main():
    x = sp.symbols('x0 x1 x2 x3')
    sp.init_printing(use_unicode=True)

    metric_tensor = conform_compacted_metric(x)

    christoffel_symbols = get_christoffel_symbols(metric_tensor, x)
    reimann_curvature_tensor = get_reimann_tensor(christoffel_symbols, x)
    ricci_curvature_tensor = get_ricci_tensor(reimann_curvature_tensor)
    raised_ricci_tensor = raise_one_index(ricci_curvature_tensor, metric_tensor)
    curvature_scalar = get_curvature_scalar(raised_ricci_tensor)

    einstein_tensor = ricci_curvature_tensor - curvature_scalar * metric_tensor
    print('\n\nEinstein Tensor: \n\n', einstein_tensor)

    filename = "EinsteinFieldEquations"

    # Build lines (avoid full LaTeX matrices)
    lines = []
    lines.append("Important Symbols and Tensors")
    lines.append("Metric tensor g_{mu nu}:")
    lines += _matrix_plain("g", metric_tensor)

    lines.append("Non‑zero Christoffel Symbols:")
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expr = christoffel_symbols[i][j][k]
                if expr == 0:
                    continue
                lines.append("$\\Gamma^{" + str(i) + "}_{" + str(j) + str(k) + "} = " + _sanitize_latex(expr) + "$")

    lines.append("Ricci Tensor R_{mu nu}:")
    lines += _matrix_plain("R", ricci_curvature_tensor)

    lines.append("Curvature Scalar:")
    lines.append("$R = " + _sanitize_latex(curvature_scalar) + "$")

    lines.append("Einstein Tensor G_{mu nu}:")
    lines += _matrix_plain("G", einstein_tensor)

    # Render to PNG
    fig_height = max(2, 0.45 * len(lines))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis('off')

    for idx, line in enumerate(lines):
        y = 1 - (idx + 1) / (len(lines) + 1)
        # mathtext only for lines starting and ending with $
        if line.startswith("$") and line.endswith("$"):
            ax.text(0.02, y, line, fontsize=11, va='top')
        else:
            ax.text(0.02, y, line, fontsize=10, va='top', family="monospace")

    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if os.path.exists(out_path):
        print(f"PNG written: {out_path}")
        os.system(f'start \"\" \"{out_path}\"')
    else:
        print("Failed to create PNG.")

if __name__ == "__main__":
    main()
