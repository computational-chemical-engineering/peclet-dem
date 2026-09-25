#!/usr/bin/env bash
# p-q-s chain (ghost_band mode margin) without the one-thread pin: 8 runs per (np, threads).
# Usage: docs/momentum_evidence/run_chain.sh <build-dir> > chain.txt
B=${1:-build_mom}/tests/kokkos_mpi/test_ghost_band_mpi
export OMP_PROC_BIND=false
for np in 2 4; do
  for thr in 8 1; do
    for rep in 1 2 3 4 5 6 7 8; do
      line=$(OMP_NUM_THREADS=$thr timeout 300 mpirun --bind-to none -np $np "$B" margin 2>&1 | grep '\[margin')
      echo "CHAIN np=$np thr=$thr rep=$rep $line"
    done
  done
done
echo "EXIT chain"
