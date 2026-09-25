#!/usr/bin/env bash
# The convergence premise (G8): one run per (mode, np) at OMP_NUM_THREADS=1; the MOMENTUM line
# carries ovl = max over steps of the global position-loop residual.
# Usage: docs/momentum_evidence/run_ovl.sh <build-dir> > ovl.txt
B=${1:-build_mom}/tests/kokkos_mpi/test_momentum_mpi
export OMP_PROC_BIND=false OMP_NUM_THREADS=1
for mode in ${MODES:-cluster cluster_friction cluster_pgs cluster_posonly hertz cluster_sync3 cluster_norot cluster_periodic}; do
  for np in 1 2 4 8; do
    line=$(timeout 600 mpirun --bind-to none -np $np "$B" $mode 2>&1 | grep '^MOMENTUM')
    echo "${line:-MOMENTUM mode=$mode np=$np ERROR}"
  done
done
echo "EXIT ovl"
