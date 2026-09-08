"""Write the unit hollow-cylinder shape as an STL for Ovito (README "LAMMPS + STL").

Ovito scales the mesh by the particle radius, so the mesh is the UNIT shape: outer radius 1, and the
height / thickness of the simulation shape divided by its radius. The SDF is sampled on a grid by a
one-particle Simulation (get_sdf_grid, the same reconstruction the packed-bed export uses) and
meshed with marching cubes.

    PYTHONPATH=build python examples/generate_particles.py [particle_shape.stl]
"""
import sys

import numpy as np
from skimage import measure
from stl import mesh

from peclet import dem

# Simulation shape (radius, height, thickness) -> unit shape relative to radius 1.
R_SIM, H_SIM, T_SIM = 0.7, 1.4, 0.3


def unit_shape_sdf(radius, height, thickness, resolution=128, margin=1.5):
    """(grid, origin, spacing): SDF of the shape centred at the origin, identity rotation, scale 1."""
    sim = dem.Simulation(1)
    sim.initialize_shape(shape_type=2, radius=radius, height=height, thickness=thickness)
    bound = max(height, 2.0 * radius) * margin / 2.0
    sim.set_domain((-bound, -bound, -bound), (bound, bound, bound))
    sim.set_positions(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    sim.set_scales(np.array([1.0], dtype=np.float32))
    sim.set_quaternions(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    grid = np.asarray(sim.get_sdf_grid((resolution, resolution, resolution)))
    spacing = 2.0 * bound / np.array(grid.shape)
    return grid, np.array([-bound, -bound, -bound]), spacing


def generate_base_shape_stl(output_file="particle_shape.stl", resolution=128):
    r_unit, h_unit, t_unit = 1.0, H_SIM / R_SIM, T_SIM / R_SIM
    print(f"Sampling the unit hollow cylinder (r={r_unit}, h={h_unit:.3f}, t={t_unit:.3f}) "
          f"on a {resolution}^3 grid...")
    grid, origin, spacing = unit_shape_sdf(r_unit, h_unit, t_unit, resolution)
    print("Running marching cubes...")
    verts, faces, _, _ = measure.marching_cubes(grid, level=0.0)
    verts_world = origin + verts * spacing
    obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype), remove_empty_areas=False)
    obj.vectors = verts_world[faces]
    obj.save(output_file)
    print(f"Saved binary STL ({faces.shape[0]} triangles) to {output_file}")


if __name__ == "__main__":
    generate_base_shape_stl(sys.argv[1] if len(sys.argv) > 1 else "particle_shape.stl")
