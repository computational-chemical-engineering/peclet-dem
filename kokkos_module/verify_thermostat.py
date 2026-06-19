import numpy as np, sys, gc
sys.path.insert(0,"build_module"); import demgpu_kokkos as dk
# free particles with random velocities + Berendsen thermostat -> translational temperature -> Ttarget.
N=200; L=50.0; R=0.5; Ttar=2.0; tau=0.05; dt=1e-3
s=dk.Simulation(N); s.set_sphere_shape(R); s.set_domain(L,L,L,False,False,False)
s.set_gravity(0,0,0); s.set_dt(dt); s.set_material_params(0,0,0); s.set_solver_iterations(4,2)
rng=np.random.default_rng(0)
pos=(rng.random((N,3))*0.8*L+0.1*L).astype(np.float32)
s.set_positions(pos)
s.set_velocities((rng.standard_normal((N,3))*5.0).astype(np.float32))  # hot start
s.set_inv_mass(np.ones(N,np.float32))
s.set_thermostat(Ttar, tau, 1.0)
def temp():
    v=s.get_velocities(); ke=0.5*np.sum(v*v); return 2.0*ke/(3.0*N*1.0)  # T = 2KE/(ndof*kB)
T0=temp()
for _ in range(400): s.step(1)
T1=temp()
print(f"thermostat: T0={T0:.3f} -> T1={T1:.3f} (target {Ttar})  err={abs(T1-Ttar)/Ttar:.2e}")
print("PASS" if abs(T1-Ttar)/Ttar < 0.1 else "FAIL")
del s; gc.collect()
