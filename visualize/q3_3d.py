import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""
Q3 3D Visualization (English Version)
Models:
1. Optimal model: Survey lines parallel to depth contours (i.e. along Y while slope is along X). Swath width is uniform (rectangles) and maximized.
2. Oblique model: A single oblique survey line crossing the slope at an angle (e.g. 45°). Swath width varies along track (trapezoidal / distorted footprint) leading to reduced effective area per unit length.

The script provides separate 3D figures instead of a combined comparison figure.
"""
# ================== Tunable Parameters ==================
THETA = 120            # Multibeam opening angle (deg)
ALPHA = 1.5            # Seabed slope (deg) slope direction along +X (depth becomes shallower toward +X after sign fix below)
H0 = 110               # Depth at x=0 (m)
LINE_LENGTH = 3704     # Along-track length shown (m) => 3.7 km (Y range: -1852 ~ 1852)
# 使用Q3 Notebook贪心算法结果的35条最优测线位置 (西深东浅)
OPT_LINE_X_POSITIONS = [
    -3361.0, -2776.7, -2237.4, -1739.9, -1280.8, -857.2, -466.3, -105.6,
     227.3,  534.4,   817.8,   1079.2,  1320.5,  1543.2,  1748.6,  1938.2,
    2113.1, 2274.5,  2423.5,  2560.9,  2687.7,  2804.7,  2912.7,  3012.3,
    3104.2, 3189.1,  3267.3,  3339.6,  3406.2,  3467.7,  3524.5,  3576.8,
    3625.1, 3669.7,  3681.7
]
# 边界 (便于构建海底网格覆盖完整区域)
X_BOUND = (-3704.0, 3704.0)
OBLIQUE_BETA_DEG = 45  # Oblique line heading angle to +X (deg); 90° means optimal direction
OBLIQUE_CENTER = (0, 0)  # Center of oblique line (x0, y0)
SAMPLES_ALONG = 40      # Samples along oblique line to construct footprint
SHOW_EDGE_WALL = True   # Show beam edge side walls for volumetric perception
SWATH_ALPHA = 0.55      # Swath footprint face alpha
EDGE_ALPHA = 0.35       # Side wall alpha
COLORMAP = 'viridis'    # Colormap for depth shading
FIGSIZE = (14, 9)
ELEV_AZIM = (28, -55)   # (elevation, azimuth) viewing angles
SAVE_FIG = False         # Set True to save figures
OUTPUT_PREFIX = 'q3_model'  # File prefix if saving

# ================== Basic Functions ==================
def to_rad(deg: float) -> float:
    return math.radians(deg)

def depth_at_x(x: float) -> float:
    """Planar seabed varying only with X.
    Notebook结果: 西端(-3704m)深度~207m, 中心(0) 110m, 东端(+3704m) ~13m.
    因此采用 Z = H0 - x * sin(ALPHA) (相对原脚本符号取反) 以匹配西深东浅。
    """
    return H0 - x * math.sin(to_rad(ALPHA))

def coverage_width(depth: float, slope_angle: float) -> float:
    """Coverage width given depth and effective cross-track slope angle (in radians).
    Formula consistent with the notebook: W = D * (1/cos(θ/2+α_eff) + 1/cos(θ/2-α_eff)) * sin(θ/2)
    当 α_eff=0 时近似 W ≈ 2D tan(θ/2) => 系数 ~3.464, 与Notebook k≈3.4724一致。
    """
    theta = to_rad(THETA)
    a = slope_angle
    term1 = 1 / math.cos(theta/2 + a)
    term2 = 1 / math.cos(theta/2 - a)
    return depth * (term1 + term2) * math.sin(theta/2)

# ================== Optimal (Parallel) Swaths ==================

def build_optimal_swath_polygons():
    polys = []
    infos = []
    # Track direction along Y => heading β=90° => effective slope α_eff = α * cos(β) = 0
    alpha_eff = 0.0
    for x0 in OPT_LINE_X_POSITIONS:
        depth_center = depth_at_x(x0)
        W = coverage_width(depth_center, alpha_eff)
        y0, y1 = -LINE_LENGTH/2, LINE_LENGTH/2
        x_left, x_right = x0 - W/2, x0 + W/2
        z_left, z_right = depth_at_x(x_left), depth_at_x(x_right)
        verts = [
            (x_left, y0, z_left),
            (x_right, y0, z_right),
            (x_right, y1, z_right),
            (x_left, y1, z_left)
        ]
        polys.append(verts)
        infos.append(dict(type='optimal', x=x0, depth=depth_center, width=W))
    return polys, infos

# ================== Oblique Swath ==================

def build_oblique_swath_polygon():
    beta_rad = to_rad(OBLIQUE_BETA_DEG)
    # Effective slope component normal to track direction
    alpha_eff = to_rad(ALPHA) * math.cos(beta_rad)
    x_c, y_c = OBLIQUE_CENTER
    t_vals = np.linspace(-LINE_LENGTH/2, LINE_LENGTH/2, SAMPLES_ALONG)
    ux, uy = math.cos(beta_rad), math.sin(beta_rad)  # Along-track unit vector
    nx, ny = -uy, ux  # Normal (cross-track) vector

    left_edge, right_edge = [], []
    widths, depths_center = [], []

    for t in t_vals:
        xc = x_c + t * ux
        yc = y_c + t * uy
        Dc = depth_at_x(xc)
        Wc = coverage_width(Dc, alpha_eff)
        xl = xc - (Wc/2) * nx
        yl = yc - (Wc/2) * ny
        xr = xc + (Wc/2) * nx
        yr = yc + (Wc/2) * ny
        zl = depth_at_x(xl)
        zr = depth_at_x(xr)
        left_edge.append((xl, yl, zl))
        right_edge.append((xr, yr, zr))
        widths.append(Wc)
        depths_center.append(Dc)

    polygon = left_edge + right_edge[::-1]
    info = dict(type='oblique', beta_deg=OBLIQUE_BETA_DEG, alpha_eff_deg=math.degrees(alpha_eff),
                width_min=min(widths), width_max=max(widths),
                depth_min=min(depths_center), depth_max=max(depths_center))
    return polygon, left_edge, right_edge, info

# ================== Seabed Mesh ==================

def build_seabed_mesh(nx=220, ny=90):
    # 覆盖到边界并留少量缓冲
    x_min = X_BOUND[0] - 80
    x_max = X_BOUND[1] + 80
    X = np.linspace(x_min, x_max, nx)
    Y = np.linspace(-LINE_LENGTH/2, LINE_LENGTH/2, ny)
    XX, YY = np.meshgrid(X, Y, indexing='xy')
    ZZ = depth_at_x(XX)
    return XX, YY, ZZ

# ================== Plot: Optimal Model ==================

def plot_optimal_model():
    polys, infos = build_optimal_swath_polygons()
    XX, YY, ZZ = build_seabed_mesh()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Q3 Optimal Survey Lines (Parallel to Contours)')

    ax.plot_surface(XX, YY, ZZ, cmap=cm.terrain, alpha=0.55, linewidth=0, antialiased=True)

    depths = np.array([i['depth'] for i in infos])
    norm = plt.Normalize(depths.min(), depths.max())
    cmap_obj = plt.get_cmap(COLORMAP)

    for verts, info in zip(polys, infos):
        color = cmap_obj(norm(info['depth']))
        poly = Poly3DCollection([verts], facecolor=color, edgecolor='k', linewidths=0.5, alpha=SWATH_ALPHA)
        ax.add_collection3d(poly)
        x0 = info['x']
        # Surface track
        ax.plot([x0, x0], [-LINE_LENGTH/2, LINE_LENGTH/2], [0, 0], color='black', linewidth=1.2)
        # Depth line at center
        ax.plot([x0, x0], [0, 0], [0, info['depth']], color='red', linewidth=0.9)
        ax.text(x0, 0, info['depth'] + 4, f"x={x0:.0f}\nW={info['width']:.1f} m", ha='center', va='bottom', fontsize=6)
        if SHOW_EDGE_WALL:
            (xl, yl, zl), (xr, yr, zr) = verts[0], verts[1]
            left_wall = Poly3DCollection([[(x0, -LINE_LENGTH/2, 0), (x0, LINE_LENGTH/2, 0), (xl, LINE_LENGTH/2, zl), (xl, -LINE_LENGTH/2, zl)]],
                                         facecolor=(color[0], color[1], color[2], EDGE_ALPHA), edgecolor='none')
            right_wall = Poly3DCollection([[(x0, -LINE_LENGTH/2, 0), (x0, LINE_LENGTH/2, 0), (xr, LINE_LENGTH/2, zr), (xr, -LINE_LENGTH/2, zr)]],
                                          facecolor=(color[0], color[1], color[2], EDGE_ALPHA), edgecolor='none')
            ax.add_collection3d(left_wall)
            ax.add_collection3d(right_wall)

    from matplotlib.cm import ScalarMappable
    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(depths)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label('Center Depth (m)')

    ax.set_xlabel('X (Across-track) / m')
    ax.set_ylabel('Y (Along-track) / m')
    ax.set_zlabel('Depth / m')
    ax.set_title('Q3 Multibeam: Optimal Lines (Greedy Result, 35 lines)', pad=18)

    # Invert depth axis (positive downward)
    z_west = depth_at_x(X_BOUND[0])
    z_east = depth_at_x(X_BOUND[1])
    z_max = max(z_west, z_east) * 1.05  # deepest * 1.05
    ax.set_zlim(z_max, 0)
    ax.set_xlim(X_BOUND[0] - 50, X_BOUND[1] + 50)
    ax.set_ylim(-LINE_LENGTH/2, LINE_LENGTH/2)

    ax.grid(True, alpha=0.3)
    ax.view_init(*ELEV_AZIM)

    ax.text2D(0.02, 0.02,
              f"Parameters: θ={THETA}°, α={ALPHA}°, H0={H0} m\nLines: {len(OPT_LINE_X_POSITIONS)}; Domain X[{X_BOUND[0]:.0f},{X_BOUND[1]:.0f}] m",\
              transform=ax.transAxes, fontsize=9,\
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.75))

    if SAVE_FIG:
        fig.savefig(f"{OUTPUT_PREFIX}_optimal.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUTPUT_PREFIX}_optimal.pdf", bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# ================== Plot: Oblique Model ==================

def plot_oblique_model():
    polygon, left_edge, right_edge, info = build_oblique_swath_polygon()
    XX, YY, ZZ = build_seabed_mesh()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Q3 Oblique Survey Line (Non-Parallel)')

    ax.plot_surface(XX, YY, ZZ, cmap=cm.terrain, alpha=0.55, linewidth=0, antialiased=True)

    # Color based on mean center depth
    depth_mean = np.mean([p[2] for p in left_edge])
    norm = plt.Normalize(depth_mean - 5, depth_mean + 5)
    cmap_obj = plt.get_cmap(COLORMAP)
    color = cmap_obj(norm(depth_mean))

    ob_poly = Poly3DCollection([polygon], facecolor=color, edgecolor='k', linewidths=0.6, alpha=0.75)
    ax.add_collection3d(ob_poly)

    # Track line on sea surface
    beta_rad = to_rad(OBLIQUE_BETA_DEG)
    ux, uy = math.cos(beta_rad), math.sin(beta_rad)
    x_c, y_c = OBLIQUE_CENTER
    y0, y1 = -LINE_LENGTH/2, LINE_LENGTH/2
    P0 = (x_c + y0 * ux, y0 * uy)
    P1 = (x_c + y1 * ux, y1 * uy)
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [0, 0], color='darkorange', linewidth=3.0, label=f"Oblique line β={OBLIQUE_BETA_DEG}°")

    # Side walls (optional)
    if SHOW_EDGE_WALL:
        left_surface_proj = [(p[0], p[1], 0) for p in left_edge]
        right_surface_proj = [(p[0], p[1], 0) for p in right_edge]
        wall_left = Poly3DCollection([left_surface_proj + left_edge[::-1]], facecolor=(1, 0.4, 0, 0.28), edgecolor='none')
        wall_right = Poly3DCollection([right_surface_proj + right_edge[::-1]], facecolor=(1, 0.4, 0, 0.28), edgecolor='none')
        ax.add_collection3d(wall_left)
        ax.add_collection3d(wall_right)

    ax.text(P1[0], P1[1], 5,
            f"Width range: {info['width_min']:.0f}–{info['width_max']:.0f} m\nα_eff≈{info['alpha_eff_deg']:.2f}°",
            fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

    ax.set_xlabel('X (Across-track) / m')
    ax.set_ylabel('Y (Along-track) / m')
    ax.set_zlabel('Depth / m')
    ax.set_title('Q3 Multibeam: Oblique Line (Non-Parallel to Contours)', pad=18)

    z_west = depth_at_x(X_BOUND[0])
    z_east = depth_at_x(X_BOUND[1])
    z_max = max(z_west, z_east) * 1.05
    ax.set_zlim(z_max, 0)
    ax.set_xlim(X_BOUND[0] - 50, X_BOUND[1] + 50)
    ax.set_ylim(-LINE_LENGTH/2, LINE_LENGTH/2)

    ax.grid(True, alpha=0.3)
    ax.view_init(*ELEV_AZIM)

    ax.text2D(0.02, 0.02,
              f"Parameters: θ={THETA}°, α={ALPHA}°, β={OBLIQUE_BETA_DEG}°\nVariable swath width due to non-zero α_eff.",
              transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.75))

    ax.legend(loc='upper right')

    if SAVE_FIG:
        fig.savefig(f"{OUTPUT_PREFIX}_oblique.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{OUTPUT_PREFIX}_oblique.pdf", bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# ================== Main ==================
if __name__ == '__main__':
    plot_optimal_model()
    plot_oblique_model()
