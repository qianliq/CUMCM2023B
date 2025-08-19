import matplotlib.pyplot as plt
import numpy as np

# Create figure with four subplots (2x2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Common parameters
ship_height = 0
water_surface = 0
seabed_depth = -50
x_range = np.linspace(-60, 60, 100)
slope = 0.3  # Seabed slope for tilted cases

# Subplot 1: Traditional Single Beam - Flat Seabed
ax1.set_title('Traditional Single Beam - Flat Seabed', fontsize=12, fontweight='bold')

# Draw water surface
ax1.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw ship
ship_x = 0
ax1.plot([ship_x-5, ship_x+5], [ship_height, ship_height], 'k-', linewidth=3, label='Survey Ship')

# Draw single beam
beam_x = [0, 0]
beam_y = [ship_height, seabed_depth]
ax1.plot(beam_x, beam_y, 'r--', linewidth=2, label='Single Beam')

# Draw flat seabed
flat_seabed_y = np.full_like(x_range, seabed_depth)  # Flat seabed
ax1.plot(x_range, flat_seabed_y, 'brown', linewidth=2, label='Seabed')

# Add measurement point
ax1.plot(0, seabed_depth, 'ro', markersize=8, label='Measurement Point')

# Fill water area
ax1.fill_between(x_range, water_surface, flat_seabed_y, alpha=0.3, color='lightblue', label='Water')

# Add vertical line from ship to seabed
ax1.plot([0, 0], [ship_height, seabed_depth], color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Ship Position')

ax1.set_xlim(-60, 60)
ax1.set_ylim(-70, 10)
ax1.set_xlabel('Horizontal Distance (m)', fontsize=10)
ax1.set_ylabel('Depth (m)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=8)

# Subplot 2: Multi-beam Sounding - Flat Seabed
ax2.set_title('Multi-beam Sounding - Flat Seabed', fontsize=12, fontweight='bold')

# Draw water surface
ax2.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw ship
ax2.plot([ship_x-5, ship_x+5], [ship_height, ship_height], 'k-', linewidth=3, label='Survey Ship')

# Draw flat seabed
ax2.plot(x_range, flat_seabed_y, 'brown', linewidth=2, label='Seabed')

# Multi-beam parameters (120 degrees opening angle, ±60 degrees)
opening_angle = 120  # degrees total
half_angle = 60  # degrees from center
beam_angles = [-60, 0, 60]  # Left edge, center, right edge
beam_colors = ['red', 'green', 'blue']
beam_widths = [2.5, 2, 2.5]

# Draw main beams for flat seabed - corrected calculation
depth = abs(seabed_depth)
for i, angle in enumerate(beam_angles):
    # For flat seabed, beam angle is measured from vertical (nadir)
    # So the horizontal distance from ship to impact point is depth * tan(angle)
    # But we need to be careful about the sign and geometry
    if angle == 0:  # Center beam goes straight down
        end_x = 0
        end_y = seabed_depth
    else:
        # For side beams, calculate horizontal distance
        angle_rad = np.radians(angle)
        end_x = depth * np.tan(angle_rad)
        end_y = seabed_depth
    
    # Draw beam
    if i == 0:
        ax2.plot([0, end_x], [ship_height, end_y], '-', 
                color=beam_colors[i], linewidth=beam_widths[i], 
                alpha=0.8, label='Multi-beam Edges & Center')
    else:
        ax2.plot([0, end_x], [ship_height, end_y], '-', 
                color=beam_colors[i], linewidth=beam_widths[i], alpha=0.8)
    
    # Add measurement points for edge beams
    if i != 1:  # Don't add point for center beam
        ax2.plot(end_x, end_y, 'o', color=beam_colors[i], markersize=6)

# Draw additional beams to show fan coverage
additional_angles = [-45, -30, -15, 15, 30, 45]
for angle in additional_angles:
    angle_rad = np.radians(angle)
    end_x = depth * np.tan(angle_rad)
    end_y = seabed_depth
    
    ax2.plot([0, end_x], [ship_height, end_y], '--', 
            color='gray', linewidth=0.8, alpha=0.5)
    ax2.plot(end_x, end_y, 'o', color='gray', markersize=2, alpha=0.6)

# Fill water area
ax2.fill_between(x_range, water_surface, flat_seabed_y, alpha=0.3, color='lightblue', label='Water')

# Add coverage area - corrected for 60 degree half angle
left_x = depth * np.tan(np.radians(-60))
right_x = depth * np.tan(np.radians(60))
coverage_x = np.linspace(left_x, right_x, 50)
coverage_y = np.full_like(coverage_x, seabed_depth)
ax2.fill_between(coverage_x, coverage_y - 1, coverage_y + 1, 
                alpha=0.3, color='red', label='Coverage Area')

# Add vertical line from ship to seabed
ax2.plot([0, 0], [ship_height, seabed_depth], color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Ship Position')

ax2.set_xlim(-60, 60)
ax2.set_ylim(-70, 10)
ax2.set_xlabel('Horizontal Distance (m)', fontsize=10)
ax2.set_ylabel('Depth (m)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=8)

# Subplot 3: Traditional Single Beam - Sloped Seabed
ax3.set_title('Traditional Single Beam - Sloped Seabed', fontsize=12, fontweight='bold')

# Draw water surface
ax3.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw ship
ax3.plot([ship_x-5, ship_x+5], [ship_height, ship_height], 'k-', linewidth=3, label='Survey Ship')

# Draw single beam
ax3.plot(beam_x, beam_y, 'r--', linewidth=2, label='Single Beam')

# Draw sloped seabed
seabed_y = seabed_depth + slope * x_range  # Sloped seabed
ax3.plot(x_range, seabed_y, 'brown', linewidth=2, label='Seabed')

# Add measurement point
ax3.plot(0, seabed_depth, 'ro', markersize=8, label='Measurement Point')

# Fill water area
ax3.fill_between(x_range, water_surface, seabed_y, alpha=0.3, color='lightblue', label='Water')

# Add vertical line from ship to seabed
ax3.plot([0, 0], [ship_height, seabed_depth], color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Ship Position')

ax3.set_xlim(-60, 60)
ax3.set_ylim(-70, 10)
ax3.set_xlabel('Horizontal Distance (m)', fontsize=10)
ax3.set_ylabel('Depth (m)', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', fontsize=8)

# Add annotations
ax3.annotate('Single point\nmeasurement', xy=(0, seabed_depth), xytext=(20, -30),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, ha='center')

# Subplot 4: Multi-beam Sounding - Sloped Seabed
ax4.set_title('Multi-beam Sounding - Sloped Seabed', fontsize=12, fontweight='bold')

# Draw water surface
ax4.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw ship
ax4.plot([ship_x-5, ship_x+5], [ship_height, ship_height], 'k-', linewidth=3, label='Survey Ship')

# Draw sloped seabed
ax4.plot(x_range, seabed_y, 'brown', linewidth=2, label='Seabed')

# Draw main beams for sloped seabed
for i, angle in enumerate(beam_angles):
    angle_rad = np.radians(angle)
    
    # For sloped seabed, find intersection between beam and seabed
    # Beam from ship (0, 0) with angle: x*tan(angle_rad) + y = 0 (y = -x*tan(angle_rad))
    # Seabed line: y = seabed_depth + slope * x
    # Intersection: -x*tan(angle_rad) = seabed_depth + slope*x
    # Rearrange: x*(-tan(angle_rad) - slope) = seabed_depth
    # x = seabed_depth / (-tan(angle_rad) - slope)
    
    denominator = -np.tan(angle_rad) - slope
    if abs(denominator) > 1e-6:
        end_x = seabed_depth / denominator
        end_y = seabed_depth + slope * end_x
        
        if -100 <= end_x <= 100 and end_y <= 0:  # Within reasonable bounds and below water
            if i == 0:
                ax4.plot([0, end_x], [ship_height, end_y], '-', 
                        color=beam_colors[i], linewidth=beam_widths[i], 
                        alpha=0.8, label='Multi-beam Edges & Center')
            else:
                ax4.plot([0, end_x], [ship_height, end_y], '-', 
                        color=beam_colors[i], linewidth=beam_widths[i], alpha=0.8)
            
            if i != 1:
                ax4.plot(end_x, end_y, 'o', color=beam_colors[i], markersize=6)

# Draw additional beams for sloped seabed
for angle in additional_angles:
    angle_rad = np.radians(angle)
    denominator = -np.tan(angle_rad) - slope
    
    if abs(denominator) > 1e-6:
        end_x = seabed_depth / denominator
        end_y = seabed_depth + slope * end_x
        
        if -100 <= end_x <= 100 and end_y <= 0:
            ax4.plot([0, end_x], [ship_height, end_y], '--', 
                    color='gray', linewidth=0.8, alpha=0.5)
            ax4.plot(end_x, end_y, 'o', color='gray', markersize=2, alpha=0.6)

# Fill water area
ax4.fill_between(x_range, water_surface, seabed_y, alpha=0.3, color='lightblue', label='Water')

# Add coverage area for sloped seabed
left_denom = -np.tan(np.radians(-60)) - slope
right_denom = -np.tan(np.radians(60)) - slope

if abs(left_denom) > 1e-6 and abs(right_denom) > 1e-6:
    left_x = seabed_depth / left_denom
    right_x = seabed_depth / right_denom
    
    coverage_x = np.linspace(min(left_x, right_x), max(left_x, right_x), 50)
    coverage_y = seabed_depth + slope * coverage_x
    ax4.fill_between(coverage_x, coverage_y - 1, coverage_y + 1, 
                    alpha=0.3, color='red', label='Coverage Area')

# Add vertical line from ship to seabed
ax4.plot([0, 0], [ship_height, seabed_depth], color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Ship Position')

ax4.set_xlim(-60, 60)
ax4.set_ylim(-70, 10)
ax4.set_xlabel('Horizontal Distance (m)', fontsize=10)
ax4.set_ylabel('Depth (m)', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/qadg/Project/CUMCM2023B/beam_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
