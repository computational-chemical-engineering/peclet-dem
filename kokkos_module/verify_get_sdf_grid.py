#!/usr/bin/env python3
# Validate the Kokkos get_sdf_grid (Eikonal SDF reconstruction) against CUDA demgpu on a FIXED ring config.
# The splat (atomic-min) + Jacobi-Eikonal are deterministic, so identical particles must give the same SDF.
# The two modules can't co-import (both own Kokkos init), so each backend runs as a subprocess writing a .npy;
# the driver compares. Usage: no args = driver; `--backend {cuda,kokkos} <out.npy>` = generate.
import os, sys, subprocess
import numpy as np

L, RES, R, H, T = 4.0, 48, 0.5, 0.75, 0.18
POS = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 3.0], [3.0, 1.0, 3.0],
                [2.0, 2.0, 2.0]], dtype=np.float32)
N = POS.shape[0]
_rng = np.random.default_rng(7)
QUAT = _rng.normal(0, 1, (N, 4)).astype(np.float32); QUAT /= np.linalg.norm(QUAT, axis=1, keepdims=True)


def gen_cuda(out):
    sys.path.insert(0, "/home/frankp/Codes/suite/packing-gpu/build")
    import demgpu
    s = demgpu.Simulation(N)
    s.initialize(shape_type=2, radius=R, height=H, thickness=T)
    s.set_domain((0.0, 0.0, 0.0), (L, L, L)); s.enable_periodicity(True, True, True)
    p4 = np.concatenate([POS, np.ones((N, 1), np.float32)], axis=1); s.set_positions(p4)
    s.set_quaternions(QUAT); s.set_scales(np.ones(N, np.float32))
    np.save(out, np.asarray(s.get_sdf_grid((RES, RES, RES)), np.float32))


def gen_kokkos(out):
    sys.path.insert(0, "/home/frankp/Codes/suite/packing-gpu/build_module")
    import demgpu_kokkos as dem, gc
    s = dem.Simulation(N * 8)
    s.initialize_shape(2, R, H, T)
    s.set_domain(L, L, L, True, True, True)
    s.set_positions(POS); s.set_quaternions(QUAT); s.set_scales(np.ones(N, np.float32))
    np.save(out, np.asarray(s.get_sdf_grid((RES, RES, RES)), np.float32))
    del s; gc.collect()


def main():
    here = os.path.abspath(__file__)
    cuda_npy, kok_npy = "/tmp/sdf_cuda.npy", "/tmp/sdf_kok.npy"
    for be, out in (("cuda", cuda_npy), ("kokkos", kok_npy)):
        p = subprocess.run([sys.executable, here, "--backend", be, out], capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout); print(p.stderr, file=sys.stderr); print(f"FAIL: {be} subprocess"); return 1
    a, b = np.load(cuda_npy), np.load(kok_npy)
    # geometry agreement: SIGN (solid vs fluid) and near-surface distance are what the CFD uses.
    sign_match = float((np.sign(a) == np.sign(b)).mean())
    band = np.abs(a) < 0.2  # near-surface band (the cut-cell region)
    band_max = float(np.abs(a[band] - b[band]).max()) if band.any() else 0.0
    full_max = float(np.abs(a - b).max())
    porosity_c, porosity_k = float((a > 0).mean()), float((b > 0).mean())
    print(f"  CUDA porosity={porosity_c:.4f}  Kokkos porosity={porosity_k:.4f}")
    print(f"  sign agreement = {sign_match*100:.3f}%   near-surface band max|Δ| = {band_max:.3e}   full max|Δ| = {full_max:.3e}")
    ok = sign_match > 0.9999 and band_max < 1e-4 and abs(porosity_c - porosity_k) < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--backend" in sys.argv:
        be = sys.argv[sys.argv.index("--backend") + 1]; out = sys.argv[-1]
        (gen_cuda if be == "cuda" else gen_kokkos)(out)
        sys.exit(0)
    sys.exit(main())
