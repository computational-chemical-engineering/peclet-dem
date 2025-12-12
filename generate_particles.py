import numpy as np
from skimage import measure
from stl import mesh

import demgpu

def hollow_cylinder_sdf_wrapper(x, y, z):
    # This wrapper function is called by the grid generator.
    # It must return the SDF value for the point (x,y,z).
    # Since existing logic accepts grids (X,Y,Z), we should vectorize this.
    # But C++ binding is scalar. 
    # For marching cubes grid generation, it's often faster to just loop or vectorise if we exposed vector version.
    # Let's write a simple vectorized wrapper around the scalar binding.
    
    # Flatten input
    shape = x.shape
    xf = x.ravel()
    yf = y.ravel()
    zf = z.ravel()
    
    res = np.zeros_like(xf)
    
    # Params: r_outer=0.7, h=1.4, thick=0.3 (r_inner=0.4 => thick=0.3)
    # Note: Previous python implementation derived r_inner=0.4.
    # r_outer = 0.7. thick = r_outer - r_inner = 0.3.
    # Params for C++: (r_outer, h, thick, unused)
    params = (0.7, 1.4, 0.3, 0.0)
    
    # We can iterate. Ideally we would expose a vectorized version in C++ (array in, array out).
    # But given resolution=128^3 = 2M points, a loop in python calling a single C++ function 2M times is slow.
    # It's better than duplicating logic but performance might suffer.
    # However, for 128^3 it might take a few seconds. Acceptable for offline tool.
    
    for i in range(len(xf)):
        res[i] = demgpu.sdf_hollow_cylinder((xf[i], yf[i], zf[i]), params)
        
    return res.reshape(shape)

def generate_base_shape_stl(output_file="particle_shape.stl", resolution=128):
    # 1. Grid generation (-1.1 to 1.1 to cover unit sphere with margin)
    vals = np.linspace(-1.1, 1.1, resolution)
    X, Y, Z = np.meshgrid(vals, vals, vals)
    
    # 2. Evaluate SDF
    print("Evaluating SDF using C++ binding...")
    # vol = sdf_func(X, Y, Z) 
    # We call our wrapper
    vol = hollow_cylinder_sdf_wrapper(X, Y, Z)
    
    # 3. Marching Cubes
    print("Running Marching Cubes...")
    # level=0.0 is the surface
    verts, faces, normals, values = measure.marching_cubes(vol, level=0.0)
    
    # 4. Create Mesh
    obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            # Scale coordinates back to physical space
            # marching_cubes returns indices. We map index to vals.
            # vals[index] approx
            # Better: use spacing and offset from measure.marching_cubes if available, 
            # OR manually map: coords = vals[0] + verts * (vals[1]-vals[0])
            pt_idx = f[j]
            # verts contains coordinates in grid index space (float)
            
            # spacing
            step = vals[1] - vals[0]
            origin = vals[0]
            
            real_pt = origin + verts[f[j]] * step
            obj.vectors[i][j] = real_pt
            
    # 5. Save Binary
    obj.save(output_file)
    print(f"Saved binary STL to {output_file}")

if __name__ == "__main__":
    generate_base_shape_stl("particle_shape.stl")
