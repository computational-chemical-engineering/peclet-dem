"""Multilevel-stabilization pile (test_colored_gs run_coloring_valid scene): dump the state after
n steps per stabilization mode to <out>/<mode>.npz, for np.array_equal between two builds."""
import sys, os
import numpy as np
from peclet import dem
out = sys.argv[1]; nsteps = int(sys.argv[2]); os.makedirs(out, exist_ok=True)
for mode in ["multilevel", "escalate", "ordered", "onesided"]:
    rng = np.random.default_rng(7)
    L = 12.0; rp = 0.5; N = 900
    Pp = np.c_[rng.uniform(1, L - 1, (N, 2)), rng.uniform(1, L - 1, N)].astype(np.float32)
    s = dem.Simulation(N + 64); s.initialize_shape('sphere', radius=rp)
    s.set_domain((0, 0, 0), (L, L, L)); s.set_periodic(False, False, False)
    s.add_plane((0, 0, 0), (0, 0, 1))
    s.set_gravity((0, 0, -9.81)); s.set_material_params(0.4, 0.3, 0.3)
    s.set_solver_iterations(12, 8); s.set_dt(2e-3)
    (s.diagnostics.set_stabilization if mode in ("escalate", "ordered") else s.set_stabilization)(mode)
    s.set_positions(np.c_[Pp, np.ones(N, np.float32)])
    s.set_velocities(np.zeros((N, 3), np.float32))
    for i in range(nsteps):
        s.step()
    np.savez(os.path.join(out, mode + ".npz"), x=s.get_positions(), v=s.get_velocities(), w=s.get_angular_velocities())
    ml = s.diagnostics.multilevel_coloring_conflicts() if hasattr(s.diagnostics, "multilevel_coloring_conflicts") else -1
    print(mode, "conflicts", s.diagnostics.coloring_conflicts(), "ml", ml)
