# Einstein Field Equations Generator (EFE_Gen.py)

Generates key geometric objects from a chosen 4D metric using SymPy:
- Christoffel symbols Γ^i_{jk}
- Riemann curvature tensor R^i_{jkl}
- Ricci tensor R_{μν}
- Raised Ricci tensor R^μ_{ ν} (one index raised)
- Curvature (Ricci) scalar R
- Einstein tensor G_{μν} = R_{μν} − R g_{μν}

Outputs a PNG file (EinsteinFieldEquations.png) summarizing non‑zero Christoffel symbols and tensors in plain text with light math formatting (Matplotlib mathtext only for simple inline expressions).

## Requirements
Install (e.g. in a virtual environment):
```
pip install sympy numpy matplotlib
```

## Run
From the project subdirectory:
```
python EFE_Gen.py
```
PNG is written next to the script.

## Selecting a Metric
Edit the line in main():
```python
metric_tensor = conform_compacted_metric(x)
```
Available helper metrics:
- conform_compacted_metric
- spherical_metric
- FRW_metric

Each expects a tuple/list of 4 SymPy symbols (x0,x1,x2,x3).

## File Structure
- EFE_Gen.py : computation + PNG export
- EinsteinFieldEquations.png : generated summary (overwritten each run)

## Implementation Notes
- Dimension fixed at 4.
- Indices assumed (0,1,2,3).
- Curvature scalar currently computed as sum of raised Ricci components (R = g^{μν} R_{μν} via one raised index). For clarity you can replace with double contraction using metric inverse if modified.
- Output omits full LaTeX matrices (mathtext subset limitations).
- Zero Christoffel symbols are skipped.

## Extending
- Replace metric function or add new ones.
- Add caching if performance becomes an issue.
- To export full LaTeX, integrate a TeX generator (not included to avoid pdflatex dependency).

## Limitations
- No signature checks.
- No simplification strategy beyond SymPy `simplify`.
- No validation of metric (symmetry, invertibility assumed).

## License
Add appropriate licensing text here if needed.
