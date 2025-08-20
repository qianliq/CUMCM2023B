import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================== Parameters (modifiable) ==================
# Multibeam total opening angle (deg)
THETA = 120
# Seabed slope along X (deg)
ALPHA = 1.5
# Water depth at center line (m)
D0 = 70
# Survey line spacing (m)
LINE_SPACING = 200
# Must be odd for symmetry (e.g. -800..800)
N_LINES = 9
# Along-track length shown (m)
LINE_LENGTH = 800
# Show overlap areas between adjacent swaths on seabed
SHOW_OVERLAP_PATCH = True
# Show two edge surfaces (beam boundary planes) for each line
SHOW_EDGE_SURFACES = True
# Seabed footprint polygon transparency
SWATH_ALPHA = 0.45
# Edge plane transparency
EDGE_SURFACE_ALPHA = 0.8  # Edge plane transparency (5%)
# Edge color
EDGE_COLOR = 'dodgerblue'
# 是否绘制海面上测线投影
SHOW_SURFACE_TRACKS = True
# 海面测线加粗线宽
SURFACE_TRACK_LINEWIDTH = 2.4
# 默认颜色（若想与条带对应可设为 None 表示继承条带颜色）
SURFACE_TRACK_COLOR = 'black'
# ============================================================

# ============== 基本函数（与 q1 模型保持一致） ==============
def to_rad(deg: float) -> float:
    return math.radians(deg)

def depth_at_x(x: float) -> float:
    """给定横向位置 x (Across-track)，返回水深 D(x)。"""
    return D0 - x * math.tan(to_rad(ALPHA))

def coverage_width(D: float) -> float:
    """给定水深 D，计算该位置的多波束覆盖宽度 W(D)。"""
    theta_rad = to_rad(THETA)
    alpha_rad = to_rad(ALPHA)
    sin_half = math.sin(theta_rad / 2)
    cos_half = math.cos(theta_rad / 2)
    tan_alpha = math.tan(alpha_rad)
    term1 = 1 / (cos_half + sin_half * tan_alpha)
    term2 = 1 / (cos_half - sin_half * tan_alpha)
    return D * sin_half * (term1 + term2)

def overlap_rate(x: float, d: float) -> float | None:
    """计算位置 x 与前一条 (x-d) 测线覆盖的相对重叠率 (相对当前条带宽度)。允许返回负值表示漏测。"""
    positions = get_positions()
    if x == positions[0]:
        return None
    D_x = depth_at_x(x)
    D_prev = depth_at_x(x - d)
    theta_rad = to_rad(THETA)
    alpha_rad = to_rad(ALPHA)
    sin_half = math.sin(theta_rad / 2)
    cos_half = math.cos(theta_rad / 2)
    tan_alpha = math.tan(alpha_rad)
    cos_alpha = math.cos(alpha_rad)
    A = cos_half + sin_half * tan_alpha
    B = cos_half - sin_half * tan_alpha
    eta = (B / A) + (D_prev / D_x) - (d / (D_x * sin_half * cos_alpha))
    return eta  # 比例 (非百分数)

# ============== 几何构建 ==============

def get_positions():
    half = N_LINES // 2
    return [ (i - half) * LINE_SPACING for i in range(N_LINES) ]


def build_seabed_mesh(nx: int = 120, ny: int = 40):
    """生成海底平面网格 (X,Y,Z)。海底只随 X 倾斜。"""
    positions = get_positions()
    x_min, x_max = min(positions), max(positions)
    X = np.linspace(x_min - LINE_SPACING, x_max + LINE_SPACING, nx)
    Y = np.linspace(-LINE_LENGTH/2, LINE_LENGTH/2, ny)
    XX, YY = np.meshgrid(X, Y, indexing='xy')
    ZZ = depth_at_x(XX)
    return XX, YY, ZZ


def swath_polygon_vertices(x_center: float, length: float, width: float):
    """Return four seabed footprint vertices (x,y,z) of a swath rectangle."""
    # 条带横向覆盖范围 (Across-track)
    x_left = x_center - width / 2
    x_right = x_center + width / 2
    # 沿航向 (Along-track) 范围
    y0 = -length / 2
    y1 = length / 2
    # 海底是倾斜平面 z = D0 - x * tan(alpha)
    z_left = depth_at_x(x_left)
    z_right = depth_at_x(x_right)
    # 四个角点 (顺时针)
    verts = [
        (x_left, y0, z_left),
        (x_right, y0, z_right),
        (x_right, y1, z_right),
        (x_left, y1, z_left),
    ]
    return verts

# 新增：构建测线边界线与侧面

def swath_edge_lines_and_surfaces(x_center: float, length: float, width: float):
    """Return data for left/right edge polylines and edge surfaces (from surface to seabed)."""
    x_left = x_center - width / 2
    x_right = x_center + width / 2
    y0, y1 = -length / 2, length / 2
    # Seabed depths at left/right (constant along Y because slope only in X)
    z_left = depth_at_x(x_left)
    z_right = depth_at_x(x_right)
    # Polylines (two points each sufficient for a straight line)
    left_line = [(x_left, y0, z_left), (x_left, y1, z_left)]
    right_line = [(x_right, y0, z_right), (x_right, y1, z_right)]
    # Edge surfaces (quadrilaterals) connecting sea surface (z=0 at x_center) to seabed edge
    # We approximate vessel track at x_center along y; edges define a vertical/oblique plane
    left_surface = [
        (x_center, y0, 0),
        (x_center, y1, 0),
        (x_left, y1, z_left),
        (x_left, y0, z_left)
    ]
    right_surface = [
        (x_center, y0, 0),
        (x_center, y1, 0),
        (x_right, y1, z_right),
        (x_right, y0, z_right)
    ]
    return left_line, right_line, left_surface, right_surface


def compute_swath_data():
    data = []
    for x in get_positions():
        D = depth_at_x(x)
        W = coverage_width(D)
        ov = overlap_rate(x, LINE_SPACING)
        data.append({
            'x': x,
            'depth': D,
            'width': W,
            'overlap_with_prev': ov  # 比例 (None / 正 / 负)
        })
    return data

# ============== 可视化 ==============

def plot_3d_swaths():
    swaths = compute_swath_data()
    XX, YY, ZZ = build_seabed_mesh()

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Multibeam Equal-Spaced Lines 3D View')

    # 1. Seabed surface
    seabed = ax.plot_surface(XX, YY, ZZ, cmap=cm.terrain, alpha=0.55, linewidth=0, antialiased=True)

    # 颜色映射依据水深（更深更蓝）
    depths = np.array([s['depth'] for s in swaths])
    norm = plt.Normalize(depths.min(), depths.max())
    cmap = plt.get_cmap('viridis')  # 替换弃用的 cm.get_cmap

    poly_collection = []

    for i, s in enumerate(swaths):
        verts = swath_polygon_vertices(s['x'], LINE_LENGTH, s['width'])
        base_color = cmap(norm(s['depth']))  # 基础颜色（RGBA）
        poly = Poly3DCollection([verts], facecolor=base_color, edgecolor='k', alpha=SWATH_ALPHA, linewidths=0.6)
        ax.add_collection3d(poly)
        poly_collection.append(poly)

        # 中心测线（从海面到海底）
        ax.plot([s['x'], s['x']], [0, 0], [0, s['depth']], color='red', linewidth=1.2)

        # 标注
        ax.text(s['x'], 0, s['depth'] + 2, f"x={s['x']} m\nW={s['width']:.1f} m",
                ha='center', va='bottom', fontsize=8, color='black', zorder=10)

        if SHOW_EDGE_SURFACES:
            left_line, right_line, left_surface, right_surface = swath_edge_lines_and_surfaces(s['x'], LINE_LENGTH, s['width'])
            # 使用底部 footprint 原色，仅降低透明度（不再暗化）
            edge_surface_color = (base_color[0], base_color[1], base_color[2], EDGE_SURFACE_ALPHA)
            edge_line_color = (base_color[0], base_color[1], base_color[2], 1.0)
            ax.plot([p[0] for p in left_line], [p[1] for p in left_line], [p[2] for p in left_line],
                    color=edge_line_color, linewidth=1.1)
            ax.plot([p[0] for p in right_line], [p[1] for p in right_line], [p[2] for p in right_line],
                    color=edge_line_color, linewidth=1.1)
            edge_poly_left = Poly3DCollection([left_surface], facecolor=edge_surface_color, edgecolor='none')
            edge_poly_right = Poly3DCollection([right_surface], facecolor=edge_surface_color, edgecolor='none')
            ax.add_collection3d(edge_poly_left)
            ax.add_collection3d(edge_poly_right)

        # 新增：海面上加粗测线（沿航向）
        if SHOW_SURFACE_TRACKS:
            surface_color = (base_color[0], base_color[1], base_color[2], 1.0) if SURFACE_TRACK_COLOR is None else SURFACE_TRACK_COLOR
            ax.plot([s['x'], s['x']], [-LINE_LENGTH/2, LINE_LENGTH/2], [0, 0],
                    color=surface_color, linewidth=SURFACE_TRACK_LINEWIDTH, solid_capstyle='round', zorder=15)

    # 2. 可选：重叠区域 (使用相邻两个条带交集)
    if SHOW_OVERLAP_PATCH:
        for i in range(1, len(swaths)):
            prev_s = swaths[i-1]
            curr_s = swaths[i]
            # 计算条带在 X 方向的区间
            prev_interval = (prev_s['x'] - prev_s['width']/2, prev_s['x'] + prev_s['width']/2)
            curr_interval = (curr_s['x'] - curr_s['width']/2, curr_s['x'] + curr_s['width']/2)
            left = max(prev_interval[0], curr_interval[0])
            right = min(prev_interval[1], curr_interval[1])
            if right > left:  # 存在重叠
                y0, y1 = -LINE_LENGTH/2, LINE_LENGTH/2
                z_left = depth_at_x(left)
                z_right = depth_at_x(right)
                ov_verts = [
                    (left, y0, z_left),
                    (right, y0, z_right),
                    (right, y1, z_right),
                    (left, y1, z_left)
                ]
                ov_poly = Poly3DCollection([ov_verts], facecolor=(1, 0.55, 0, 0.35), edgecolor='none')
                ax.add_collection3d(ov_poly)

    # 3. Axes & labels (English)
    ax.set_xlabel('Across-track X (m)', labelpad=10)
    ax.set_ylabel('Along-track Y (m)', labelpad=10)
    ax.set_zlabel('Depth (m)', labelpad=10)
    ax.set_title('Multibeam Survey 3D Coverage (Equal Spacing)\nSlope along X; rectangles are seabed footprints', pad=20)

    # 深度向下: 通过反转 Z 轴
    zmax = max(depths) * 1.15
    ax.set_zlim(zmax, 0)  # 上 0, 下 正

    # X / Y 范围
    positions = get_positions()
    ax.set_xlim(min(positions) - LINE_SPACING, max(positions) + LINE_SPACING)
    ax.set_ylim(-LINE_LENGTH/2, LINE_LENGTH/2)

    # 色标 (对应条带中心深度)
    from matplotlib.cm import ScalarMappable
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(depths)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)  # 指定 ax 以避免 ValueError
    cb.set_label('Center Depth (m)')

    # 网格与视角
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=28, azim=-55)

    # 辅助说明文本
    txt = (
        f"Parameters: θ={THETA}°, α={ALPHA}°, D0={D0} m, spacing={LINE_SPACING} m\n" \
        f"Lines={N_LINES}, along-track length={LINE_LENGTH} m\n" \
        f"Orange = overlap area; Blue translucent planes = beam edge surfaces"
    )
    ax.text2D(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.7))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_3d_swaths()
