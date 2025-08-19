import matplotlib.pyplot as plt
import numpy as np

# Create figure for simplified multi-beam model
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Common parameters
ship_height = 0
water_surface = 0
seabed_depth = -50
x_range = np.linspace(-80, 80, 100)
slope = 0.3  # Seabed slope

# Draw water surface
ax.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw ship
ship_x = 0
ax.plot([ship_x-8, ship_x+8], [ship_height, ship_height], 'k-', linewidth=4, label='Survey Ship')

# Draw sloped seabed
seabed_y = seabed_depth + slope * x_range
ax.plot(x_range, seabed_y, 'brown', linewidth=3, label='Seabed')

# Multi-beam parameters (120 degrees opening angle, ±60 degrees)
beam_angles = [-60, 60]  # Left and right edge beams
beam_colors = ['red', 'red']
beam_widths = [3, 3]

# Calculate beam intersections with sloped seabed
beam_endpoints = []
for i, angle in enumerate(beam_angles):
    angle_rad = np.radians(angle)
    # For sloped seabed intersection calculation
    denominator = -np.tan(angle_rad) - slope
    if abs(denominator) > 1e-6:
        end_x = seabed_depth / denominator
        end_y = seabed_depth + slope * end_x
        beam_endpoints.append((end_x, end_y))
        
        # Draw beam
        ax.plot([0, end_x], [ship_height, end_y], '-', 
                color=beam_colors[i], linewidth=beam_widths[i], 
                alpha=0.8, label='Multi-beam Edge' if i == 0 else '')
        
        # Add measurement point
        ax.plot(end_x, end_y, 'o', color='red', markersize=8)

# Fill water area
ax.fill_between(x_range, water_surface, seabed_y, alpha=0.3, color='lightblue', label='Water')

# Draw coverage area
if len(beam_endpoints) == 2:
    left_x, left_y = beam_endpoints[0]
    right_x, right_y = beam_endpoints[1]
    
    coverage_x = np.linspace(left_x, right_x, 50)
    coverage_y = seabed_depth + slope * coverage_x
    ax.fill_between(coverage_x, coverage_y - 2, coverage_y + 2, 
                    alpha=0.3, color='yellow', label='Coverage Area')
    
    # Calculate and annotate horizontal coverage width W
    W = abs(right_x - left_x)
    
    # Draw horizontal line showing W - using the actual intersection points
    y_annotation = min(left_y, right_y) - 8
    ax.plot([left_x, right_x], [y_annotation, y_annotation], 'g-', linewidth=3)
    
    # Draw vertical lines from intersection points to W line
    ax.plot([left_x, left_x], [left_y, y_annotation], 'g--', linewidth=2, alpha=0.7)
    ax.plot([right_x, right_x], [right_y, y_annotation], 'g--', linewidth=2, alpha=0.7)
    
    # Draw end markers
    ax.plot([left_x, left_x], [y_annotation-2, y_annotation+2], 'g-', linewidth=2)
    ax.plot([right_x, right_x], [y_annotation-2, y_annotation+2], 'g-', linewidth=2)
    
    # Add W annotation
    mid_x = (left_x + right_x) / 2
    ax.annotate(f'W = {W:.1f}m', xy=(mid_x, y_annotation), 
                xytext=(mid_x, y_annotation-8),
                ha='center', va='center', fontsize=14, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Add vertical line from ship to seabed at center
ax.plot([0, 0], [ship_height, seabed_depth], color='black', 
        linestyle=':', linewidth=2, alpha=0.7, label='Ship Position')

# Add depth annotation
ax.annotate('D', xy=(-5, seabed_depth/2), xytext=(-15, seabed_depth/2),
            ha='center', va='center', fontsize=14, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'))

# Add angle annotations
ax.annotate('60°', xy=(0, ship_height), xytext=(15, -5),
            ha='center', va='center', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Add slope angle annotation
slope_x = 30
slope_y = seabed_depth + slope * slope_x
ax.annotate('α', xy=(slope_x, slope_y), xytext=(slope_x+10, slope_y+5),
            ha='center', va='center', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='brown', lw=1.5))

# Set axis properties
ax.set_xlim(-80, 80)
ax.set_ylim(-70, 10)
ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Simplified Multi-beam Sounding Model with Coverage Width W', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Set equal aspect ratio (1:1 scale)
ax.set_aspect('equal', adjustable='box')

# Add parameter text box
textstr = f'Parameters:\nBeam Angle: 60°\nSlope: {np.degrees(np.arctan(slope)):.1f}°\nDepth at Center: {abs(seabed_depth)}m'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/qadg/Project/CUMCM2023B/simplified_multibeam_model.png', dpi=300, bbox_inches='tight')
plt.show()
