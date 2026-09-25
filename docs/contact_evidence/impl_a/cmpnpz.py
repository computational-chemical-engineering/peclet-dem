import sys, glob, os, numpy as np
a, b = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(a, "*.npz"))):
    A = np.load(f); B = np.load(os.path.join(b, os.path.basename(f)))
    ok = all(np.array_equal(A[k], B[k]) for k in A.files)
    print(("IDENT " if ok else "DIFF  ") + os.path.basename(f))
