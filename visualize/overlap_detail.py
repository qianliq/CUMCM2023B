import matplotlib.pyplot as plt
import numpy as np

# Create figure for adjacent survey lines overlap model
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Parameters
ship_height = 0
water_surface = 0
seabed_depth = -60
slope = 0.3  # Seabed slope
beam_half_angle = 60  # Half angle in degrees
d = 50  # Reduce distance between survey lines for larger overlap

# Survey ship positions
ship1_x = -d  # Ship 1 position (x-d)
ship2_x = 0   # Ship 2 position (x)

# X range for seabed
x_range = np.linspace(-150, 150, 500)
seabed_y = seabed_depth + slope * x_range

# Draw water surface
ax.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw seabed
ax.plot(x_range, seabed_y, 'brown', linewidth=3, label='Seabed (Slope α)')

# Draw ships
ax.plot([ship1_x-8, ship1_x+8], [ship_height, ship_height], 'k-', linewidth=4)
ax.plot([ship2_x-8, ship2_x+8], [ship_height, ship_height], 'k-', linewidth=4)

# Ship labels
ax.text(ship1_x, ship_height + 8, 'Ship 1', ha='center', va='bottom', 
        fontsize=12, fontweight='bold')
ax.text(ship2_x, ship_height + 8, 'Ship 2', ha='center', va='bottom', 
        fontsize=12, fontweight='bold')

# Calculate beam intersection points for both ships
def calculate_beam_intersection(ship_x, angle_deg):
    angle_rad = np.radians(angle_deg)
    # Correct beam intersection calculation with sloped seabed
    # Beam equation from ship: y - ship_height = -(x - ship_x) * tan(angle_rad)
    # Rearranged: y = ship_height - (x - ship_x) * tan(angle_rad)
    # Seabed equation: y = seabed_depth + slope * x
    # Set equal: ship_height - (x - ship_x) * tan(angle_rad) = seabed_depth + slope * x
    # Solve for x: ship_height - seabed_depth = (x - ship_x) * tan(angle_rad) + slope * x
    # ship_height - seabed_depth = x * tan(angle_rad) - ship_x * tan(angle_rad) + slope * x
    # ship_height - seabed_depth + ship_x * tan(angle_rad) = x * (tan(angle_rad) + slope)
    
    denominator = np.tan(angle_rad) + slope
    if abs(denominator) > 1e-6:
        end_x = (ship_height - seabed_depth + ship_x * np.tan(angle_rad)) / denominator
        end_y = seabed_depth + slope * end_x
        return end_x, end_y
    return None, None

# Ship 1 coverage boundaries
B_x, B_y = calculate_beam_intersection(ship1_x, -beam_half_angle)  # Left boundary of ship 1
right1_x, right1_y = calculate_beam_intersection(ship1_x, beam_half_angle)  # Right boundary of ship 1

# Ship 2 coverage boundaries  
left2_x, left2_y = calculate_beam_intersection(ship2_x, -beam_half_angle)  # Left boundary of ship 2
I_x, I_y = calculate_beam_intersection(ship2_x, beam_half_angle)  # Right boundary of ship 2

# Projection points G and G' (directly below ships)
G_x = ship1_x
G_y = seabed_depth + slope * G_x
Gp_x = ship2_x
Gp_y = seabed_depth + slope * Gp_x

# Overlap boundaries (H and F)
H_x = left2_x  # Left boundary of overlap (same as left boundary of ship 2)
H_y = left2_y
F_x = right1_x  # Right boundary of overlap (same as right boundary of ship 1)
F_y = right1_y

# Draw beams
# Ship 1 beams
ax.plot([ship1_x, B_x], [ship_height, B_y], 'r-', linewidth=2, alpha=0.8, label='Ship 1 Beams')
ax.plot([ship1_x, right1_x], [ship_height, right1_y], 'r-', linewidth=2, alpha=0.8)

# Ship 2 beams
ax.plot([ship2_x, left2_x], [ship_height, left2_y], 'g-', linewidth=2, alpha=0.8, label='Ship 2 Beams')
ax.plot([ship2_x, I_x], [ship_height, I_y], 'g-', linewidth=2, alpha=0.8)

# Draw angle annotations
ax.annotate('θ/2', xy=(ship1_x + 15, ship_height - 8), xytext=(ship1_x + 25, ship_height - 15),
            ha='center', va='center', fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1))

ax.annotate('θ/2', xy=(ship2_x - 15, ship_height - 8), xytext=(ship2_x - 25, ship_height - 15),
            ha='center', va='center', fontsize=11, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1))

# Mark key points on seabed
points = [
    (B_x, B_y, 'B', 'red'),
    (G_x, G_y, 'G', 'black'),
    (H_x, H_y, 'H', 'blue'),
    (F_x, F_y, 'F', 'blue'),
    (Gp_x, Gp_y, "G'", 'black'),
    (I_x, I_y, 'I', 'green')
]

for px, py, label, color in points:
    ax.plot(px, py, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    ax.text(px, py - 6, label, ha='center', va='top', fontsize=12, fontweight='bold', color=color)

# Draw coverage regions with labels
regions = [
    (B_x, H_x, 'Left 1', 'red', 0.2),
    (H_x, F_x, 'Overlap', 'yellow', 0.5),
    (F_x, I_x, 'Right 2', 'green', 0.2)
]

for start_x, end_x, region_label, color, alpha in regions:
    region_x = np.linspace(start_x, end_x, 50)
    region_y = seabed_depth + slope * region_x
    ax.fill_between(region_x, region_y - 2, region_y + 2, alpha=alpha, color=color)
    
    # Add region label
    mid_x = (start_x + end_x) / 2
    mid_y = seabed_depth + slope * mid_x
    ax.text(mid_x, mid_y + 8, region_label, ha='center', va='bottom', 
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Draw vertical dashed lines from ships to seabed
ax.plot([ship1_x, ship1_x], [ship_height, G_y], 'k--', linewidth=1, alpha=0.6)
ax.plot([ship2_x, ship2_x], [ship_height, Gp_y], 'k--', linewidth=1, alpha=0.6)

# Draw horizontal measurement lines
bottom_y = min([B_y, G_y, H_y, F_y, Gp_y, I_y]) - 12

# Distance d between ships
ax.annotate('', xy=(ship1_x, bottom_y - 5), xytext=(ship2_x, bottom_y - 5),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text((ship1_x + ship2_x)/2, bottom_y - 8, f'd = {d}m', ha='center', va='top',
        fontsize=11, fontweight='bold')

# Overlap length HF
ax.annotate('', xy=(H_x, bottom_y - 15), xytext=(F_x, bottom_y - 15),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
overlap_length = abs(F_x - H_x)
ax.text((H_x + F_x)/2, bottom_y - 18, f'HF = {overlap_length:.1f}m', ha='center', va='top',
        fontsize=11, fontweight='bold', color='blue')

# Coverage width of ship 2 (HI)
ax.annotate('', xy=(H_x, bottom_y - 25), xytext=(I_x, bottom_y - 25),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
coverage_width = abs(I_x - H_x)
ax.text((H_x + I_x)/2, bottom_y - 28, f'HI = {coverage_width:.1f}m', ha='center', va='top',
        fontsize=11, fontweight='bold', color='green')

# Calculate overlap rate
overlap_rate = overlap_length / coverage_width * 100

# Fill water area
ax.fill_between(x_range, water_surface, seabed_y, alpha=0.3, color='lightblue', label='Water')

# Add slope angle annotation
slope_angle_deg = np.degrees(np.arctan(slope))
slope_x = 80
slope_y = seabed_depth + slope * slope_x
ax.annotate(f'α = {slope_angle_deg:.1f}°', xy=(slope_x, slope_y), xytext=(slope_x + 20, slope_y + 8),
            ha='center', va='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='brown', lw=1.5))

# Set axis properties
ax.set_xlim(-200, 150)
ax.set_ylim(-100, 25)
ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Adjacent Multi-beam Survey Lines Overlap Model', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Set equal aspect ratio
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('/Users/qadg/Project/CUMCM2023B/adjacent_lines_overlap_model.png', dpi=300, bbox_inches='tight')
plt.show()
