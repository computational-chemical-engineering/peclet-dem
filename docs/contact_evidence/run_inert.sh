#!/usr/bin/env bash
# Inertness of the test edits: every pre-existing momentum mode (final-state --dump, OMP 1) and
# ownership mode (printed report) of the new binaries against binaries built from HEAD's sources.
# Usage: docs/contact_evidence/run_inert.sh <old-bin-dir> <new-build-dir> <scratch-dir>
OLD=$1; NEW=$2/tests/kokkos_mpi; T=$3
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive OMP_NUM_THREADS=1
for np in 1 4; do
  for m in cluster cluster_friction cluster_pgs cluster_posonly hertz cluster_sync3 cluster_norot cluster_periodic cluster_jacobi; do
    mpirun --bind-to none -np $np $OLD/test_momentum_mpi $m --dump=$T/o_${m}_$np.bin 2>&1 | grep '^MOMENTUM' > $T/o_${m}_$np.txt
    mpirun --bind-to none -np $np $NEW/test_momentum_mpi $m --dump=$T/n_${m}_$np.bin 2>&1 | grep '^MOMENTUM' > $T/n_${m}_$np.txt
    if cmp -s $T/o_${m}_$np.bin $T/n_${m}_$np.bin && cmp -s $T/o_${m}_$np.txt $T/n_${m}_$np.txt; then r=identical; else r=DIFFERENT; fi
    echo "INERT momentum mode=$m np=$np dump+line: $r"
  done
done
for np in 1 2 4 8; do
  for m in reverse_closed reverse_periodic exactly_once_closed exactly_once_periodic exactly_once_drift; do
    a=$(mpirun --bind-to none -np $np $OLD/test_ownership_mpi $m 2>&1); b=$(mpirun --bind-to none -np $np $NEW/test_ownership_mpi $m 2>&1)
    [ "$a" = "$b" ] && r=identical || r=DIFFERENT
    echo "INERT ownership mode=$m np=$np report: $r"
  done
done
echo "EXIT inert"
