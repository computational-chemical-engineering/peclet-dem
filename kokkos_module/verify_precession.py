import numpy as np, sys, gc
sys.path.insert(0,"build_module"); import demgpu_kokkos as dk
# free spinning rigid body with anisotropic inertia -> torque-free precession; check L & E conservation.
s=dk.Simulation(1); s.set_sphere_shape(0.5); s.set_domain(100,100,100,False,False,False)
s.set_gravity(0,0,0); s.set_dt(1e-4)
s.set_positions(np.array([[0,0,0]],np.float32))
Iinv=np.array([[1.0, 1.0/3.0, 1.0/5.0]],np.float32)  # anisotropic (Ixx=1,Iyy=3,Izz=5)
s.set_inv_inertia(Iinv); s.set_inv_mass(np.array([1.0],np.float32))
s.set_quaternions(np.array([[0,0,0,1]],np.float32))
s.set_angular_velocities(np.array([[0.2,2.0,0.1]],np.float32))  # mostly about y, perturbed -> precession
def Lmag_E(s):
    w=s.get_angular_velocities()[0]; q=s.get_quaternions()[0]; iinv=s.get_inv_inertia()[0]
    # body-frame: rotate w into body, I*w_body, back to world (L); E=0.5 w.I.w
    # invRotate: q=(x,y,z,w_); approximate L magnitude & E in BODY frame (frame-independent magnitudes)
    # use body angular velocity via quaternion conj
    x,y,z,wq=q; 
    # rotate w by conj(q): simple since we just need magnitudes; do it in world using I (anisotropic) is frame-dep.
    return w
w0=s.get_angular_velocities()[0].copy()
Es=[]; Ls=[]
for it in range(5000):
    s.step(1)
    w=s.get_angular_velocities()[0]; iinv=s.get_inv_inertia()[0]; I=1.0/iinv
    # body-frame invariants: |L|^2 = sum (I_k w_k)^2 ; 2E = sum I_k w_k^2  (w here is body-frame angVel)
    Ls.append(np.sqrt(np.sum((I*w)**2))); Es.append(np.sum(I*w*w))
Ls=np.array(Ls); Es=np.array(Es)
dL=(Ls.max()-Ls.min())/Ls.mean(); dE=(Es.max()-Es.min())/Es.mean()
print(f"precession 5000 steps: |L| drift={dL:.2e}  2E drift={dE:.2e}  (anisotropic free body)")
print("PASS" if dL<1e-2 and dE<1e-2 else "FAIL")
del s; gc.collect()
