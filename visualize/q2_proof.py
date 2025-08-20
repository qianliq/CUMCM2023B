# -*- coding: utf-8 -*-
"""
q2_proof.py

3D visualization for β = 45°: vessel sails along a survey line, showing at each measurement point
both boundary beams (left/right) and an extended semi-transparent swath plane on the seabed.

Geometry (consistent with q2):
- Seabed varies only along +X (downslope). Seabed plane: z = D0 + x * tan(alpha), z is positive downward.
- β is the angle between survey line direction and downslope direction (projection of slope normal).
- Chosen β = 45°.
- Track direction unit vector u = (cosβ, sinβ, 0)
- Cross-track unit vector v = (-sinβ, cosβ, 0)
- Vessel at sea surface (z = 0). Directly below (vertical) depth D = D0 + x * tan(alpha).
- Effective slope: alpha_eff = arctan( tan(alpha) * sin β ).
- Coverage width W (using q1 formula with effective slope):
  W = D * sin(θ/2) * [ 1/(cos(θ/2)+sin(θ/2)tan(alpha_eff)) + 1/(cos(θ/2)-sin(θ/2)tan(alpha_eff)) ]
- Seabed boundary footprint endpoints: P ± (W/2) * v (horizontal shift); seabed z recomputed with x.
- Beam edge rays are lines from ship position (x,y,0) to seabed edge (x_edge,y_edge,z_edge).

New additions per request:
1. All comments / labels translated to English.
2. For each measurement, extend the seabed boundary line across-track by a factor (ACROSS_EXTEND_FACTOR).
3. Generate a semi-transparent quadrilateral swath plane (larger than original swath) spanning along-track
   (half step before & after the point) and extended across-track. Each plane lies on the seabed surface (non-planar in general;
   we approximate with bilinear patch by assigning seabed depth to each corner).
4. Added an infinite-like extended coverage line along seabed for each measurement to emphasize it lies on a plane; factor configurable.

Adjustable parameters: θ (THETA_DEG), α (ALPHA_DEG), D0, β, total track distance, sampling interval.
Outputs:
1. 3D figure: seabed surface, track, beam edge rays, extended boundary lines, semi-transparent swath planes.
2. 2D top view: footprints (original left/right edges) for reference.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D registration)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass

# ---------------- Configuration ----------------
THETA_DEG = 120.0        # Total beam opening angle θ
ALPHA_DEG = 1.5          # Seabed slope α (along +X)
BETA_DEG = 45.0          # Survey line angle β relative to downslope (+X)
D0 = 120.0               # Reference depth at origin (m)
TOTAL_DISTANCE_NM = 2.1  # Total survey distance (nautical miles)
STEP_NM = 0.3            # Sampling spacing (nautical miles)
NAUTICAL_TO_M = 1852.0

# Visualization extent factor for seabed plane
PLANE_EXTEND_FACTOR = 1.1

# Swath extension factors
ACROSS_EXTEND_FACTOR = 1.8   # How much wider (vs real W) to draw the extended seabed line & plane
ALONG_EXTEND_RATIO = 0.5     # Half-length along-track (multiples of step spacing) for plane on each side
SWATH_PLANE_ALPHA = 0.18     # Transparency for swath planes
SWATH_PLANE_FACE_COLOR = (0.2, 0.6, 0.95)  # RGB in [0,1]
EDGE_LINE_ALPHA = 0.85
COVERAGE_LINE_EXTEND_FACTOR = 4.0  # NEW: factor to extend the core coverage line beyond real swath (half-width multiplier)
COVERAGE_LINE_ALPHA = 0.55
COVERAGE_LINE_COLOR = 'magenta'

# ---------------- Model Definition ----------------
@dataclass
class SwathModel:
    theta: float  # radians
    alpha: float  # radians (true slope)
    beta: float   # radians (track direction)
    d0: float

    def half_theta(self):
        return self.theta / 2

    def effective_slope(self):
        """Effective slope α_eff = arctan(tan α * sin β)."""
        return np.arctan(np.tan(self.alpha) * np.sin(self.beta))

    def depth(self, x):
        """Seabed plane depth (positive downward)."""
        return self.d0 + x * np.tan(self.alpha)

    def coverage_width(self, depth_value):
        """Coverage width W at a given local depth using modified q1 formula."""
        alpha_eff = self.effective_slope()
        h = self.half_theta()
        cos_h = np.cos(h)
        sin_h = np.sin(h)
        t_eff = np.tan(alpha_eff)
        denom1 = cos_h + sin_h * t_eff
        denom2 = cos_h - sin_h * t_eff
        if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
            return np.nan
        width_factor = sin_h * (1/denom1 + 1/denom2)
        return depth_value * width_factor

# ---------------- Data Generation ----------------
def generate_track_and_swaths(model: SwathModel,
                              total_distance_nm: float,
                              step_nm: float):
    """Generate track points and left/right swath edge footprints.
    Returns dict with arrays.
    """
    beta = model.beta
    u = np.array([np.cos(beta), np.sin(beta), 0.0])      # along-track
    v = np.array([-np.sin(beta), np.cos(beta), 0.0])     # cross-track

    distances_nm = np.arange(0, total_distance_nm + 1e-9, step_nm)
    distances_m = distances_nm * NAUTICAL_TO_M

    ship_pts, seabed_pts = [], []
    left_edges, right_edges, widths = [], [], []

    for r_m in distances_m:
        pos_xy = u[:2] * r_m
        x, y = pos_xy
        depth_here = model.depth(x)
        W = model.coverage_width(depth_here)
        left_xy = pos_xy + v[:2] * (-W/2)
        right_xy = pos_xy + v[:2] * (W/2)
        left_z = model.depth(left_xy[0])
        right_z = model.depth(right_xy[0])

        ship_pts.append([x, y, 0.0])
        seabed_pts.append([x, y, depth_here])
        left_edges.append([left_xy[0], left_xy[1], left_z])
        right_edges.append([right_xy[0], right_xy[1], right_z])
        widths.append(W)

    return {
        'distances_nm': distances_nm,
        'ship_pts': np.array(ship_pts),
        'seabed_pts': np.array(seabed_pts),
        'left_edges': np.array(left_edges),
        'right_edges': np.array(right_edges),
        'widths': np.array(widths),
        'u': u,
        'v': v
    }

# ---------------- Visualization ----------------
def build_swath_planes(model: SwathModel, data: dict):
    """Build extended swath plane polygons for each measurement.
    Each plane is a quadrilateral aligned along u & v, with across-track exaggerated.
    Returns list of dicts with 'verts' (list of 4 (x,y,z))."""
    ship_pts = data['ship_pts']
    seabed_pts = data['seabed_pts']
    widths = data['widths']
    u = data['u']
    v = data['v']

    # Along-track spacing estimation
    if len(ship_pts) > 1:
        dxy = np.linalg.norm(np.diff(ship_pts[:,:2], axis=0), axis=1)
        avg_spacing = float(np.mean(dxy))
    else:
        avg_spacing = STEP_NM * NAUTICAL_TO_M
    half_len = avg_spacing * ALONG_EXTEND_RATIO

    planes = []
    for i in range(len(ship_pts)):
        center = seabed_pts[i]  # (x, y, z at seabed directly below ship)
        W = widths[i]
        if not np.isfinite(W) or W <= 0:
            continue
        half_across = (W/2) * ACROSS_EXTEND_FACTOR
        # Four corners in order (counter-clockwise): back-left, back-right, front-right, front-left
        corners_local = [
            -half_len * u + (-half_across) * v,
            -half_len * u + ( half_across) * v,
             half_len * u + ( half_across) * v,
             half_len * u + (-half_across) * v,
        ]
        verts = []
        for vec in corners_local:
            x = center[0] + vec[0]
            y = center[1] + vec[1]
            z = model.depth(x)  # depth depends only on x
            verts.append((x, y, z))
        planes.append({'index': i, 'verts': verts})
    return planes

def plot_3d_swath(model: SwathModel, data: dict):
    ship_pts = data['ship_pts']
    seabed_pts = data['seabed_pts']
    left_edges = data['left_edges']
    right_edges = data['right_edges']
    # NEW: cross-track unit vector for extended coverage line
    v = data['v']

    # Seabed surface grid
    all_x = np.concatenate([ship_pts[:,0], left_edges[:,0], right_edges[:,0]])
    all_y = np.concatenate([ship_pts[:,1], left_edges[:,1], right_edges[:,1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    x_rng = x_max - x_min
    y_rng = y_max - y_min
    x_pad = x_rng * (PLANE_EXTEND_FACTOR - 1) / 2
    y_pad = y_rng * (PLANE_EXTEND_FACTOR - 1) / 2

    X = np.linspace(x_min - x_pad, x_max + x_pad, 60)
    Y = np.linspace(y_min - y_pad, y_max + y_pad, 60)
    XX, YY = np.meshgrid(X, Y)
    ZZ = model.depth(XX)

    planes = build_swath_planes(model, data)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121, projection='3d')

    # Seabed surface
    surf = ax.plot_surface(XX, YY, ZZ, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)
    plt.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label='Depth (m)')

    # Track at surface & seabed projection
    ax.plot(ship_pts[:,0], ship_pts[:,1], ship_pts[:,2], 'k--', label='Track (surface)')
    ax.plot(seabed_pts[:,0], seabed_pts[:,1], seabed_pts[:,2], 'k:', label='Track projection (seabed)')

    # Beam edges and original footprint lines
    for i in range(len(ship_pts)):
        ax.plot([ship_pts[i,0], left_edges[i,0]], [ship_pts[i,1], left_edges[i,1]], [ship_pts[i,2], left_edges[i,2]],
                color='crimson', linewidth=1.0, alpha=0.9)
        ax.plot([ship_pts[i,0], right_edges[i,0]], [ship_pts[i,1], right_edges[i,1]], [ship_pts[i,2], right_edges[i,2]],
                color='navy', linewidth=1.0, alpha=0.9)
        ax.plot([left_edges[i,0], right_edges[i,0]], [left_edges[i,1], right_edges[i,1]], [left_edges[i,2], right_edges[i,2]],
                color='orange', linewidth=1.1, alpha=EDGE_LINE_ALPHA)

    # Extended across-track lines & swath planes
    poly_collection = []
    extended_line_added_legend = False  # ensure legend entry appears once
    for plane in planes:
        verts = plane['verts']
        # Extended center line using midpoints of left/right edges (average along-track front+back)
        left_mid = ((verts[0][0]+verts[3][0])/2, (verts[0][1]+verts[3][1])/2, (verts[0][2]+verts[3][2])/2)
        right_mid = ((verts[1][0]+verts[2][0])/2, (verts[1][1]+verts[2][1])/2, (verts[1][2]+verts[2][2])/2)
        ax.plot([left_mid[0], right_mid[0]], [left_mid[1], right_mid[1]], [left_mid[2], right_mid[2]],
                color='gold', linewidth=1.4, alpha=0.7)
        poly_collection.append(verts)

    # NEW: extended coverage lines (conceptual infinite intersection line) per measurement
    for i in range(len(seabed_pts)):
        # center seabed point
        cx, cy, cz = seabed_pts[i]
        # approximate local width (distance between edges)
        W_local = np.linalg.norm(left_edges[i,:2] - right_edges[i,:2])
        if not np.isfinite(W_local) or W_local <= 0:
            continue
        half_extended = (W_local/2) * COVERAGE_LINE_EXTEND_FACTOR
        p1_xy = np.array([cx, cy]) - half_extended * v[:2]
        p2_xy = np.array([cx, cy]) + half_extended * v[:2]
        p1_z = model.depth(p1_xy[0])
        p2_z = model.depth(p2_xy[0])
        ax.plot([p1_xy[0], p2_xy[0]], [p1_xy[1], p2_xy[1]], [p1_z, p2_z],
                color=COVERAGE_LINE_COLOR, alpha=COVERAGE_LINE_ALPHA, linewidth=1.3,
                label='Extended coverage line' if not extended_line_added_legend else None)
        extended_line_added_legend = True

    swath_faces = Poly3DCollection(poly_collection, facecolors=[SWATH_PLANE_FACE_COLOR]*len(poly_collection),
                                   edgecolors='none', alpha=SWATH_PLANE_ALPHA)
    ax.add_collection3d(swath_faces)

    # Labels & view
    ax.set_xlabel('X (downslope, m)')
    ax.set_ylabel('Y (cross-track, m)')
    ax.set_zlabel('Depth z (m, downward)')
    ax.set_title(f'Multibeam 3D Coverage (β={BETA_DEG}°)')
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc='upper left')

    # 2D top view
    ax2 = fig.add_subplot(122)
    ax2.set_title('Top View (Footprints)')
    ax2.plot(seabed_pts[:,0], seabed_pts[:,1], 'k.-', label='Track projection')
    ax2.plot(left_edges[:,0], left_edges[:,1], 'r.-', label='Left edge')
    ax2.plot(right_edges[:,0], right_edges[:,1], 'b.-', label='Right edge')
    for i in range(len(seabed_pts)):
        ax2.plot([left_edges[i,0], right_edges[i,0]], [left_edges[i,1], right_edges[i,1]], color='orange', linewidth=1, alpha=0.5)
        # overlay extended line in top view (same color)
        cx, cy, cz = seabed_pts[i]
        W_local = np.linalg.norm(left_edges[i,:2] - right_edges[i,:2])
        if not np.isfinite(W_local) or W_local <= 0:
            continue
        half_extended = (W_local/2) * COVERAGE_LINE_EXTEND_FACTOR
        p1_xy = np.array([cx, cy]) - half_extended * v[:2]
        p2_xy = np.array([cx, cy]) + half_extended * v[:2]
        ax2.plot([p1_xy[0], p2_xy[0]], [p1_xy[1], p2_xy[1]], color=COVERAGE_LINE_COLOR, alpha=0.35, linewidth=1.0)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    ax2.grid(alpha=0.3)
    ax2.legend()

    info = (f"θ={THETA_DEG}°  α={ALPHA_DEG}°  β={BETA_DEG}°\n"
            f"Samples: {len(ship_pts)}  Track: {TOTAL_DISTANCE_NM} nm\n"
            f"Across extend: x{ACROSS_EXTEND_FACTOR}  Along half-step ratio: {ALONG_EXTEND_RATIO}\n"
            f"Coverage line extend factor: x{COVERAGE_LINE_EXTEND_FACTOR}")
    ax2.text(0.02, 0.98, info, transform=ax2.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

# ---------------- Entry Point ----------------
def main():
    theta = np.radians(THETA_DEG)
    alpha = np.radians(ALPHA_DEG)
    beta = np.radians(BETA_DEG)
    model = SwathModel(theta=theta, alpha=alpha, beta=beta, d0=D0)

    data = generate_track_and_swaths(model, TOTAL_DISTANCE_NM, STEP_NM)
    print(f"Generated {len(data['ship_pts'])} samples; mean width: {np.nanmean(data['widths']):.2f} m")
    print(f"Width range: {np.nanmin(data['widths']):.2f} ~ {np.nanmax(data['widths']):.2f} m")
    plot_3d_swath(model, data)

if __name__ == '__main__':
    main()
