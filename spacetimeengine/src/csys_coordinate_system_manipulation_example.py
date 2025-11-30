# CSYS rotation visualization - to scale
# Requirements: matplotlib, numpy, ipywidgets
# In a Jupyter notebook, run: 
#   %pip install ipywidgets matplotlib numpy
# Then: 
#   from ipywidgets import interact, FloatSlider, Checkbox
#   (widgets are used below)

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, FloatText, Checkbox

# --- Core math: rotation matrix ---
def R(theta_deg):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Proper 2D rotation matrix
    return np.array([[c, -s],
                     [s,  c]])

# --- Helper: draw axes and vectors ---
def draw_arrow(ax, origin, vec, color, label=None, width=0.015):
    ax.arrow(origin[0], origin[1], vec[0], vec[1], 
             head_width=0.12, head_length=0.18, 
             length_includes_head=True, color=color, linewidth=2)
    if label:
        end = origin + vec
        ax.text(end[0] + 0.1, end[1] + 0.1, label, color=color, fontsize=11)

def plot_csys(theta_deg=30.0, ax_len=1.0, vx=1.0, vy=0.0, show_components=True, show_grid=True):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', adjustable='box')  # Preserve scale fidelity
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Limits based on vector and axes
    limit = max(1.5*ax_len, 1.5*abs(vx), 1.5*abs(vy), 2.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.3)

    # Original axes (black)
    i = np.array([ax_len, 0.0])
    j = np.array([0.0, ax_len])
    draw_arrow(ax, np.zeros(2), i, color='black', label='X')
    draw_arrow(ax, np.zeros(2), j, color='black', label='Y')

    # Rotated axes (orange)
    Rm = R(theta_deg)
    i_p = Rm @ i
    j_p = Rm @ j
    draw_arrow(ax, np.zeros(2), i_p, color='orange', label="X'")
    draw_arrow(ax, np.zeros(2), j_p, color='orange', label="Y'")

    # Vector in original frame (blue)
    v = np.array([vx, vy])
    draw_arrow(ax, np.zeros(2), v, color='royalblue', label='a')

    # Same geometric vector, represented in rotated coordinates (green):
    # Components of v in rotated basis are v' = R(-theta) v, but the drawn arrow remains at the same endpoint.
    v_rot_coords = R(-theta_deg) @ v  # coordinates of v in rotated CSYS
    # For visual clarity, we can optionally reconstruct v from rotated components in rotated basis:
    # v_reconstructed = v_rot_coords[0] * i_p/ax_len + v_rot_coords[1] * j_p/ax_len  # equals v
    draw_arrow(ax, np.zeros(2), v, color='seagreen', label="a (same vector)")

    # Magnitudes
    mag_v = np.linalg.norm(v)
    ax.text(-limit+0.1, limit-0.4, f"|a| = {mag_v:.3f}", fontsize=11, color='royalblue')

    # Angle annotation
    # Draw arc from X to X' for theta
    arc_r = 0.8*ax_len
    ts = np.linspace(0, np.deg2rad(theta_deg), 50)
    ax.plot(arc_r*np.cos(ts), arc_r*np.sin(ts), color='orange', linewidth=2)
    ax.text(arc_r*np.cos(np.deg2rad(theta_deg/2))+0.1, 
            arc_r*np.sin(np.deg2rad(theta_deg/2))+0.1, 
            f"θ = {theta_deg:.2f}°", color='orange', fontsize=11)

    # Component projections (optional)
    if show_components:
        # Projections onto rotated axes (green dashed from v to rotated axes)
        # Compute scalars a1' and a2' such that v = a1' e1' + a2' e2'
        a1p, a2p = v_rot_coords
        # Endpoints of component arrows along rotated axes
        comp1_end = (a1p/ax_len) * i_p  # scale basis by coefficient
        comp2_end = (a2p/ax_len) * j_p
        # Draw dashed guides
        ax.plot([0, comp1_end[0]], [0, comp1_end[1]], color='seagreen', linestyle='--', linewidth=1.8)
        ax.plot([comp1_end[0], v[0]], [comp1_end[1], v[1]], color='seagreen', linestyle='--', linewidth=1.8)
        ax.plot([0, comp2_end[0]], [0, comp2_end[1]], color='seagreen', linestyle='--', linewidth=1.8)
        ax.plot([comp2_end[0], v[0]], [comp2_end[1], v[1]], color='seagreen', linestyle='--', linewidth=1.8)

        ax.text(comp1_end[0]+0.1, comp1_end[1]+0.1, f"a₁' = {a1p:.3f}", color='seagreen', fontsize=11)
        ax.text(comp2_end[0]+0.1, comp2_end[1]+0.1, f"a₂' = {a2p:.3f}", color='seagreen', fontsize=11)

    # Legend proxies
    ax.plot([], [], color='black', label='Original CSYS (X, Y)')
    ax.plot([], [], color='orange', label="Rotated CSYS (X', Y')")
    ax.plot([], [], color='royalblue', label='Vector a (original coords)')
    ax.plot([], [], color='seagreen', label='Vector a (same geometry)')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.show()

# Interactive controls
interact(
    plot_csys,
    theta_deg=FloatSlider(value=30.0, min=-180.0, max=180.0, step=0.5, description='Rotation θ (deg)'),
    ax_len=FloatSlider(value=1.0, min=0.5, max=3.0, step=0.1, description='Axis length'),
    vx=FloatText(value=1.0, description='a_x'),
    vy=FloatText(value=0.0, description='a_y'),
    show_components=Checkbox(value=True, description='Show components in rotated frame'),
    show_grid=Checkbox(value=True, description='Grid')
);
