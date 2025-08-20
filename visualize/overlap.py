import matplotlib.pyplot as plt
import numpy as np

# Create figure for multi-beam survey lines
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Common parameters
ship_height = 0
water_surface = 0
seabed_depth = -70
x_range = np.linspace(-200, 200, 1000)
slope = np.radians(3.0)  # Increase slope to 3.0 degrees

# Draw water surface
ax.axhline(y=water_surface, color='blue', linewidth=2, label='Water Surface')

# Draw sloped seabed
seabed_y = seabed_depth - slope * x_range  # Negative slope (west deep, east shallow)
ax.plot(x_range, seabed_y, 'brown', linewidth=3, label='Seabed')

# Multi-beam parameters (120 degrees opening angle, ±60 degrees)
beam_half_angle = 60  # degrees
beam_angle_rad = np.radians(beam_half_angle)

# Survey lines positions (multiple ships) - reduce spacing for denser coverage
survey_lines = [-120, -60, 0, 60, 120]  # x positions of survey lines, spacing reduced to 60m
colors = ['red', 'green', 'orange', 'purple', 'blue']
coverage_areas = []

# Process each survey line
for i, ship_x in enumerate(survey_lines):
    # Draw ship
    ax.plot([ship_x-6, ship_x+6], [ship_height, ship_height], 'k-', linewidth=3)
    
    # Calculate beam intersections with sloped seabed
    beam_endpoints = []
    beam_angles = [-beam_half_angle, beam_half_angle]  # Left and right edge beams
    
    for angle in beam_angles:
        angle_rad = np.radians(angle)
        
        # Correct beam intersection calculation with sloped seabed
        # Beam equation from ship: y - ship_height = -(x - ship_x) * tan(angle_rad)
        # Rearranged: y = ship_height - (x - ship_x) * tan(angle_rad)
        # Seabed equation: y = seabed_depth - slope * x
        # Set equal: ship_height - (x - ship_x) * tan(angle_rad) = seabed_depth - slope * x
        # Solve for x: ship_height - seabed_depth = (x - ship_x) * tan(angle_rad) - slope * x
        # ship_height - seabed_depth = x * tan(angle_rad) - ship_x * tan(angle_rad) - slope * x
        # ship_height - seabed_depth + ship_x * tan(angle_rad) = x * (tan(angle_rad) - slope)
        
        denominator = np.tan(angle_rad) - slope
        if abs(denominator) > 1e-6:
            end_x = (ship_height - seabed_depth + ship_x * np.tan(angle_rad)) / denominator
            end_y = seabed_depth - slope * end_x
            beam_endpoints.append((end_x, end_y))
    
    # Draw beams and coverage
    if len(beam_endpoints) == 2:
        left_x, left_y = beam_endpoints[0]
        right_x, right_y = beam_endpoints[1]
        
        # Draw beam lines
        ax.plot([ship_x, left_x], [ship_height, left_y], '-', 
                color=colors[i], linewidth=2, alpha=0.8)
        ax.plot([ship_x, right_x], [ship_height, right_y], '-', 
                color=colors[i], linewidth=2, alpha=0.8)
        
        # Draw measurement points
        ax.plot([left_x, right_x], [left_y, right_y], 'o', 
                color=colors[i], markersize=6)
        
        # Store coverage area
        coverage_areas.append((left_x, right_x, left_y, right_y, colors[i]))
        
        # Draw coverage area on seabed
        coverage_x = np.linspace(left_x, right_x, 50)
        coverage_y = seabed_depth - slope * coverage_x
        ax.fill_between(coverage_x, coverage_y - 1, coverage_y + 1, 
                        alpha=0.2, color=colors[i])
        
        # Calculate and display coverage width
        W = abs(right_x - left_x)
        mid_x = (left_x + right_x) / 2
        mid_y = (left_y + right_y) / 2
        ax.text(mid_x, mid_y - 5, f'W={W:.0f}m', ha='center', va='top',
                fontsize=10, fontweight='bold', color=colors[i],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Fill water area
ax.fill_between(x_range, water_surface, seabed_y, alpha=0.3, color='lightblue', label='Water')

# Highlight overlap areas
for i in range(len(coverage_areas) - 1):
    left_x1, right_x1, left_y1, right_y1, color1 = coverage_areas[i]
    left_x2, right_x2, left_y2, right_y2, color2 = coverage_areas[i + 1]
    
    # Check for overlap
    overlap_left = max(left_x1, left_x2)
    overlap_right = min(right_x1, right_x2)
    
    if overlap_left < overlap_right:  # There is overlap
        overlap_x = np.linspace(overlap_left, overlap_right, 30)
        overlap_y = seabed_depth - slope * overlap_x
        ax.fill_between(overlap_x, overlap_y - 2, overlap_y + 2, 
                        alpha=0.5, color='yellow', 
                        label='Overlap Area' if i == 0 else '')

# Add survey line labels
for i, ship_x in enumerate(survey_lines):
    ax.text(ship_x, ship_height + 3, f'Line {i+1}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=colors[i])

# Add slope angle annotation
slope_x = 100
slope_y = seabed_depth - slope * slope_x
ax.annotate('α=3.0°', xy=(slope_x, slope_y), xytext=(slope_x+30, slope_y+8),
            ha='center', va='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='brown', lw=1.5))

# Add depth annotation
center_depth = seabed_depth - slope * 0
ax.annotate(f'D₀={abs(center_depth):.0f}m', xy=(0, center_depth/2), xytext=(-30, center_depth/2),
            ha='center', va='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'))

# Set axis properties
ax.set_xlim(-200, 200)
ax.set_ylim(-90, 15)
ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Multi-beam Survey Lines with Coverage and Overlap Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Set equal aspect ratio
ax.set_aspect('equal', adjustable='box')

# Add parameter text box
textstr = 'Parameters:\nBeam Angle: 120°\nSlope: 3.0°\nSurvey Lines: 5\nSpacing: 60m'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/qadg/Project/CUMCM2023B/multi_beam_survey_lines.png', dpi=300, bbox_inches='tight')
plt.show()
ax.set_aspect('equal', adjustable='box')

# Add parameter text box
textstr = 'Parameters:\nBeam Angle: 120°\nSlope: 3.0°\nSurvey Lines: 5\nSpacing: 60m'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/qadg/Project/CUMCM2023B/multi_beam_survey_lines.png', dpi=300, bbox_inches='tight')
plt.show()
