import numpy as np
import math

def generate_hollow_cylinder_points(radius, height, thickness, spacing):
    """
    Generates points on the surface of a hollow cylinder (Raschig Ring).
    
    Args:
        radius (float): Outer radius (R)
        height (float): Height (h)
        thickness (float): Wall thickness (t)
        spacing (float): Approximate spacing between points based on grid/resolution
    
    Returns:
        np.array: (N, 4) array of points (x, y, z, 0)
    """
    points = []
    
    r_outer = radius
    r_inner = radius - thickness
    h = height
    
    # 1. Outer Surface
    circumference = 2.0 * math.pi * r_outer
    n_angular = math.ceil(circumference / spacing)
    n_vertical = math.ceil(h / spacing)
    
    for i in range(n_angular):
        theta = 2.0 * math.pi * i / n_angular
        for j in range(n_vertical + 1):
            y = -0.5 * h + h * j / n_vertical
            x = r_outer * math.cos(theta)
            z = r_outer * math.sin(theta)
            points.append([x, y, z, 0.0])

    # 2. Inner Surface (if thick)
    if thickness > 0.0:
        circumference = 2.0 * math.pi * r_inner
        n_angular = math.ceil(circumference / spacing)
        for i in range(n_angular):
            theta = 2.0 * math.pi * i / n_angular
            for j in range(n_vertical + 1):
                y = -0.5 * h + h * j / n_vertical
                x = r_inner * math.cos(theta)
                z = r_inner * math.sin(theta)
                points.append([x, y, z, 0.0])
                
    # 3. Caps (Annulus)
    dr = spacing
    n_radial = math.ceil(thickness / dr)
    for i in range(n_radial + 1):
        r = r_inner + (r_outer - r_inner) * i / n_radial
        circumference = 2.0 * math.pi * r
        n_angular = math.ceil(circumference / spacing)
        for j in range(n_angular):
            theta = 2.0 * math.pi * j / n_angular
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            
            # Top
            points.append([x, 0.5 * h, z, 0.0])
            # Bottom
            points.append([x, -0.5 * h, z, 0.0])
            
    return np.array(points, dtype=np.float32)

if __name__ == "__main__":
    # Test
    pts = generate_hollow_cylinder_points(0.5, 2.0, 0.2, 0.1)
    print(f"Generated {len(pts)} points")
