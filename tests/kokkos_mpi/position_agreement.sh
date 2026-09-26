#!/bin/bash
# WO-12 gate (docs/contact_solve_framework.md §13.5 WO-12 acceptance 2): the accumulated,
# retractable position projection has a unique fixed point, so the converged positions of one
# overlap-only substep agree across rank counts: max |x_npN - x_np1| <= 1e-4 R.
# usage: position_agreement.sh <mpiexec> <preflags...> -- <test_momentum_mpi> <np> <outdir>
#                              [<mode> [<mode arguments...>]]
# The mode defaults to cluster_posonly --steps=1 (WO-12). ring_collide_posonly (4 steps of the
# overlap-free tube scene, docs/contact_physics_followups.md §3.4 / G-B2) extends the claim to
# multi-point units over several steps: per-step uniqueness plus Lipschitz propagation (R-B8).
MPI=$1; shift
PRE=()
while [ "$1" != "--" ]; do PRE+=("$1"); shift; done
shift
BIN=$1; NP=$2; OUT=$3
shift 3
if [ $# -eq 0 ]; then set -- cluster_posonly --steps=1; fi
mkdir -p "$OUT"
ARGS=("$@" --no-stop --pos-iters=2000)
"$MPI" -np 1 "${PRE[@]}" "$BIN" "${ARGS[@]}" --dump="$OUT/np1.dump" > "$OUT/np1.log" 2>&1 || exit 1
"$MPI" -np "$NP" "${PRE[@]}" "$BIN" "${ARGS[@]}" --dump="$OUT/np$NP.dump" > "$OUT/np$NP.log" 2>&1 || exit 1
python3 - "$OUT/np1.dump" "$OUT/np$NP.dump" <<'PY'
import struct, sys
def load(f):
    b = open(f, "rb").read(); d = {}
    for i in range(len(b) // 56):
        r = struct.unpack("<i13f", b[56 * i:56 * i + 56]); d[r[0]] = r[1:4]
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
R = 0.5
assert a.keys() == b.keys(), "different body sets"
mx = max(max(abs(a[g][k] - b[g][k]) for k in range(3)) for g in a) / R
print(f"POSAGREE max|dx|/R = {mx:.3e} (gate 1e-4)")
sys.exit(0 if mx <= 1e-4 else 1)
PY
