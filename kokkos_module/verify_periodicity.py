import numpy as np, sys, gc
sys.path.insert(0,"build_module"); import demgpu_kokkos as dk
L=20.0; R=1.0
def run(px):
    s=dk.Simulation(8); s.set_sphere_shape(R); s.set_domain(L,L,L, px, False, False)
    s.set_gravity(0,0,0); s.set_dt(1e-3); s.set_material_params(0,0,0); s.set_solver_iterations(10,2)
    s.set_positions(np.array([[0.5,L/2,L/2],[L-0.5,L/2,L/2]],np.float32))  # gap through wrap=1.0<2R
    s.set_inv_mass(np.ones(2,np.float32)); s.step(1); nc=s.num_contacts(); del s; gc.collect(); return nc
on=run(True); off=run(False)
print(f"periodic-x contacts={on}  non-periodic contacts={off}")
print("PASS" if on>0 and off==0 else "FAIL")
