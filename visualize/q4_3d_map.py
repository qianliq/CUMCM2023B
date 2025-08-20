import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import warnings
warnings.filterwarnings('ignore')

# ===================== 可配置参数 =====================
EXCEL_PATH = '附件.xlsx'
LEVEL_STEP = 5                # 等高线间隔 (米)
COHERENCE_WINDOW = 5          # 方向趋同度窗口大小(奇数)
N_ZONES = 4
ALPHA_SLOPE = 0.6
BETA_COHER = 0.4
GAMMA_SLOPE = 2.0
ARROW_STEP = 8                # 方向箭头降采样步长
SURFACE_ALPHA = 0.92
ELEV_AZIM = (35, -55)
SAVE_FIG = False
OUTPUT_PREFIX = 'q4_3d'
FIG_DPI = 160
FONT_FAMILY = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False

# ===================== 数据加载与预处理 =====================

def load_depth_data(path=EXCEL_PATH):
    data = pd.read_excel(path)
    # 列解释方式与 q4graph.ipynb 保持一致
    x_coords = data.iloc[0, 2:].values.astype(float)  # 东西方向 (海里)
    y_coords = data.iloc[1:, 1].values.astype(float)  # 南北方向 (海里)
    depth = data.iloc[1:, 2:].values.astype(float)    # 深度 (m)
    X, Y = np.meshgrid(x_coords, y_coords)
    return X, Y, depth, x_coords, y_coords

# ===================== 核心计算函数 =====================

def compute_grad_slope_direction(depth):
    dy, dx = np.gradient(depth)
    slope = np.sqrt(dx**2 + dy**2)
    # 等高线方向(垂直梯度) 角度标准化到[0, π]
    dy_safe = np.where(np.abs(dy) < 1e-10, 1e-10, dy)
    theta = np.arctan(dx / dy_safe) + np.pi/2
    theta = np.mod(theta, np.pi)
    return dx, dy, slope, theta

def compute_direction_coherence(theta, window=COHERENCE_WINDOW):
    assert window % 2 == 1, '窗口需为奇数'
    h = window // 2
    coh = np.full_like(theta, np.nan, dtype=float)
    for i in range(h, theta.shape[0]-h):
        block_rows = slice(i-h, i+h+1)
        for j in range(h, theta.shape[1]-h):
            block = theta[block_rows, j-h:j+h+1]
            cos_mean = np.mean(np.cos(2 * block))
            sin_mean = np.mean(np.sin(2 * block))
            R = np.sqrt(cos_mean**2 + sin_mean**2)
            # 圆形标准差 (R 可能接近0)
            if R < 1e-8:
                circular_std = np.pi / np.sqrt(3)  # 极大不确定
            else:
                circular_std = np.sqrt(-2 * np.log(R))
            coh[i, j] = 1 - np.tanh(circular_std)  # 映射到(0,1)
    return coh

def compute_zoning_weight(slope, coherence, alpha=ALPHA_SLOPE, beta=BETA_COHER, gamma=GAMMA_SLOPE):
    slope_w = np.tanh(gamma * slope)
    # coherence 已在[0,1) 近似
    zoning_w = alpha * slope_w + beta * coherence
    return zoning_w, slope_w

def perform_zoning(weight, n_zones=N_ZONES):
    valid = weight[~np.isnan(weight)]
    qs = np.quantile(valid, np.linspace(0, 1, n_zones+1))
    zones = np.zeros_like(weight, dtype=int)
    for i in range(n_zones):
        if i == 0:
            mask = weight <= qs[i+1]
        elif i == n_zones - 1:
            mask = weight > qs[i]
        else:
            mask = (weight > qs[i]) & (weight <= qs[i+1])
        zones[mask] = i + 1
    return zones, qs

# ===================== 3D 绘图工具函数 =====================

def _prep_ax(title, xlabel='East-West (NM)', ylabel='North-South (NM)', zlabel='Depth (m)', elev_azim=ELEV_AZIM):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, pad=14)
    ax.view_init(*elev_azim)
    return fig, ax

def plot_depth_surface(X, Y, depth):
    fig, ax = _prep_ax('3D Seafloor Depth Surface')
    norm = colors.Normalize(vmin=np.nanmin(depth), vmax=np.nanmax(depth))
    surf = ax.plot_surface(X, Y, depth, cmap='viridis_r', norm=norm, linewidth=0, antialiased=True, alpha=SURFACE_ALPHA)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='Depth (m)')
    if SAVE_FIG:
        fig.savefig(f'{OUTPUT_PREFIX}_depth.png', dpi=FIG_DPI, bbox_inches='tight')

# ---- 新增: 带底部等高线投影与坡度高亮的增强三维展示 ----
def plot_depth_surface_enhanced(X, Y, depth, slope=None, show_wire=True):
    """增强三维地形展示: 
    1) 主体彩色深度表面
    2) 底部平面等高线投影 (帮助理解起伏)
    3) 可选网格线 / 坡度高亮(>阈值)
    """
    fig, ax = _prep_ax('Enhanced 3D Seafloor (Surface + Contour Projection)')
    dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    norm = colors.Normalize(vmin=dmin, vmax=dmax)
    surf = ax.plot_surface(X, Y, depth, cmap='viridis_r', norm=norm, linewidth=0.2 if show_wire else 0, edgecolor='k' if show_wire else 'none', antialiased=True, alpha=0.93)
    # 底部等高线投影平面
    base_offset = dmin - 8  # 在最浅之下放一平面
    levels = np.linspace(dmin, dmax, 16)
    cset = ax.contour(X, Y, depth, levels=levels, zdir='z', offset=base_offset, cmap='viridis_r', linewidths=0.8, alpha=0.85)
    # 坡度高亮 (若提供 slope)
    if slope is not None:
        thr = np.nanpercentile(slope, 85)  # 上15%作为陡峭区域
        steep_mask = (slope >= thr)
        # 稀疏采样点绘制
        step = max(1, int(max(X.shape)/120))
        Xs = X[::step, ::step][steep_mask[::step, ::step]]
        Ys = Y[::step, ::step][steep_mask[::step, ::step]]
        Zs = depth[::step, ::step][steep_mask[::step, ::step]] + 1.2  # 轻微抬升
        ax.scatter(Xs, Ys, Zs, c='crimson', s=10, alpha=0.65, depthshade=False, label='Steep (top 15%)')
    # 颜色条
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='Depth (m)')
    # 轴范围与反向: 使得视觉上“向下”更深
    ax.set_zlim(base_offset, dmax + 5)
    # 文字注释
    ax.text2D(0.02, 0.02, f'Depth range {dmin:.1f}–{dmax:.1f} m', transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.6))
    if slope is not None:
        ax.legend(loc='upper right')
    if SAVE_FIG:
        fig.savefig(f'{OUTPUT_PREFIX}_depth_enhanced.png', dpi=FIG_DPI, bbox_inches='tight')


def plot_colored_by_variable(X, Y, depth, var, title, cmap='plasma', label='Value'):
    fig, ax = _prep_ax(title)
    norm = colors.Normalize(vmin=np.nanmin(var), vmax=np.nanmax(var))
    facecolors = cm.get_cmap(cmap)(norm(var))
    ax.plot_surface(X, Y, depth, facecolors=facecolors, linewidth=0, antialiased=False, alpha=1.0, shade=False)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(var)
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08, label=label)
    if SAVE_FIG:
        safe = title.lower().replace(' ', '_').replace('/', '_')
        fig.savefig(f'{OUTPUT_PREFIX}_{safe}.png', dpi=FIG_DPI, bbox_inches='tight')


def plot_zones_3d(X, Y, depth, zones, qs):
    fig, ax = _prep_ax('3D Zone Classification on Depth Surface')
    cmap_discrete = cm.get_cmap('tab10', N_ZONES)
    zone_colors = cmap_discrete((zones - 1) % N_ZONES)
    ax.plot_surface(X, Y, depth, facecolors=zone_colors, linewidth=0, antialiased=False, shade=False)
    # 自定义图例
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cmap_discrete(i), label=f'Zone {i+1}') for i in range(N_ZONES)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.1, 0.95))
    if SAVE_FIG:
        fig.savefig(f'{OUTPUT_PREFIX}_zones.png', dpi=FIG_DPI, bbox_inches='tight')


def plot_direction_field_3d(X, Y, depth, theta, step=ARROW_STEP):
    fig, ax = _prep_ax('3D Contour Direction Field (Quiver)')
    # 基础表面 (简化采样)
    step_surf = max(1, step//2)
    ax.plot_surface(X[::step_surf, ::step_surf], Y[::step_surf, ::step_surf], depth[::step_surf, ::step_surf], cmap='Greys', alpha=0.35, linewidth=0)
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Zq = depth[::step, ::step]
    U = np.cos(theta[::step, ::step])
    V = np.sin(theta[::step, ::step])
    # 在深度面上添加微小抬升避免箭头被遮挡
    Zq_up = Zq + 0.5
    ax.quiver(Xq, Yq, Zq_up, U, V, np.zeros_like(U), length=0.3, normalize=True, color='crimson', linewidth=0.5)
    if SAVE_FIG:
        fig.savefig(f'{OUTPUT_PREFIX}_direction_field.png', dpi=FIG_DPI, bbox_inches='tight')

# ===================== q3风格三维地形图 =====================
SHOW_Q3_STYLE_TERRAIN = True
SHOW_Q3_STYLE_ZONES = True  # 新增：显示分区着色3D图
Q3_STYLE_FIGSIZE = (14, 9)
Q3_STYLE_VIEW = (28, -55)

def plot_q3_style_terrain(X, Y, depth):
    """仿 q3_3d 风格：单独海底表面 + 反转深度轴(上方为0海面)。"""
    fig = plt.figure(figsize=Q3_STYLE_FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Q4 Seafloor Terrain (q3 style)')
    # 使用 terrain 或 viridis_r 颜色映射
    surf = ax.plot_surface(X, Y, depth, cmap=cm.terrain, alpha=0.62, linewidth=0, antialiased=True)
    # z 轴: 0 为海面 (若需要可在此加一透明平面)
    z_max = depth.max() * 1.05
    ax.set_zlim(z_max, 0)
    ax.set_xlabel('X (East-West) / NM')
    ax.set_ylabel('Y (North-South) / NM')
    ax.set_zlabel('Depth / m')
    ax.set_title('Q4 Seafloor 3D Terrain (q3 style)', pad=18)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.grid(True, alpha=0.3)
    ax.view_init(*Q3_STYLE_VIEW)
    # 颜色条
    from matplotlib.cm import ScalarMappable
    norm = plt.Normalize(depth.min(), depth.max())
    mappable = ScalarMappable(norm=norm, cmap=cm.terrain)
    mappable.set_array(depth)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label('Depth (m)')
    # 文本参数
    ax.text2D(0.02, 0.02,
              f"Depth range: {depth.min():.1f}-{depth.max():.1f} m\nGrid: {depth.shape[0]}×{depth.shape[1]}",
              transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.75))
    plt.tight_layout()
    plt.show()

def plot_q3_style_zones(X, Y, depth, zones):
    """按照分区结果给地形着色 (仿 q3 风格)"""
    fig = plt.figure(figsize=Q3_STYLE_FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Q4 Seafloor Zones (q3 style)')
    # 离散调色板
    cmap_discrete = cm.get_cmap('tab10', N_ZONES)
    # 生成与 depth 相同形状的颜色数组
    zone_norm = (zones - 1) % N_ZONES
    facecolors = cmap_discrete(zone_norm)
    surf = ax.plot_surface(X, Y, depth, facecolors=facecolors, linewidth=0, antialiased=True, shade=False, alpha=0.92)
    # 叠加分区边界等高线 (在Z方向投影到海面上方或略抬升)
    boundary_levels = [i + 0.5 for i in range(1, N_ZONES)]
    ax.contour(X, Y, zones, levels=boundary_levels, colors='k', linewidths=1.0, linestyles='--', offset=None)
    z_max = depth.max() * 1.05
    ax.set_zlim(z_max, 0)
    ax.set_xlabel('X (East-West) / NM')
    ax.set_ylabel('Y (North-South) / NM')
    ax.set_zlabel('Depth / m')
    ax.set_title('Q4 Seafloor Zone Classification (3D)', pad=18)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.grid(True, alpha=0.3)
    ax.view_init(*Q3_STYLE_VIEW)
    # 图例
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cmap_discrete(i), label=f'Zone {i+1}') for i in range(N_ZONES)]
    ax.legend(handles=handles, loc='upper right')
    ax.text2D(0.02, 0.02,
              f"Zones: {N_ZONES}\nDepth {depth.min():.1f}-{depth.max():.1f} m", transform=ax.transAxes,
              fontsize=9, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.75))
    if SAVE_FIG:
        fig.savefig(f'{OUTPUT_PREFIX}_zones_q3_style.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# ===================== 主执行流程 =====================

def main():
    X, Y, depth, x_coords, y_coords = load_depth_data()
    dx, dy, slope, theta = compute_grad_slope_direction(depth)
    coherence = compute_direction_coherence(theta)
    zoning_weight, slope_weight = compute_zoning_weight(slope, coherence)
    zones, quantiles = perform_zoning(zoning_weight)

    print('Data summary:')
    print(f'  X range (NM): {x_coords.min()} ~ {x_coords.max()}')
    print(f'  Y range (NM): {y_coords.min()} ~ {y_coords.max()}')
    print(f'  Depth range (m): {np.nanmin(depth):.1f} ~ {np.nanmax(depth):.1f}')
    print(f'  Slope mean/max: {np.nanmean(slope):.3f} / {np.nanmax(slope):.3f}')
    print(f'  Coherence range: {np.nanmin(coherence):.3f} ~ {np.nanmax(coherence):.3f}')
    print(f'  Weight range: {np.nanmin(zoning_weight):.3f} ~ {np.nanmax(zoning_weight):.3f}')

    print('\nZone statistics:')
    for i in range(1, N_ZONES+1):
        mask = zones == i
        if np.any(mask):
            area_ratio = mask.sum() / np.isfinite(zones).sum() * 100
            print(f'  Zone {i}: area {area_ratio:.1f}%, slope mean {slope[mask].mean():.3f}, coherence mean {coherence[mask].mean():.3f}, weight quantile [{quantiles[i-1]:.3f},{quantiles[i]:.3f}]')

    # 绘制图形
    if SHOW_Q3_STYLE_TERRAIN:
        plot_q3_style_terrain(X, Y, depth)
    if SHOW_Q3_STYLE_ZONES:
        plot_q3_style_zones(X, Y, depth, zones)
    plot_depth_surface(X, Y, depth)
    plot_depth_surface_enhanced(X, Y, depth, slope=slope, show_wire=True)
    plot_colored_by_variable(X, Y, depth, slope, 'Slope Colored Depth Surface', cmap='Reds', label='Slope')
    plot_colored_by_variable(X, Y, depth, coherence, 'Direction Coherence Colored Depth Surface', cmap='Blues', label='Coherence')
    plot_colored_by_variable(X, Y, depth, zoning_weight, 'Zoning Weight Colored Depth Surface', cmap='plasma', label='Weight')
    plot_zones_3d(X, Y, depth, zones, quantiles)
    plot_direction_field_3d(X, Y, depth, theta)

if __name__ == '__main__':
    main()
