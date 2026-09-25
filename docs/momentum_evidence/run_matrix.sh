#!/usr/bin/env bash
# Conservation matrix: modes x np {1,2,4,8} x OMP_NUM_THREADS {1,8} x 3 repeats.
# Usage: docs/momentum_evidence/run_matrix.sh <build-dir> > matrix.txt
#        MODES="cluster_sync3 cluster_norot" REPS=1 docs/momentum_evidence/run_matrix.sh <build-dir>
B=${1:-build_mom}/tests/kokkos_mpi/test_momentum_mpi
# Passive OpenMP waiting: 8-thread ranks on a loaded shared host otherwise spin past the timeout
# (np 8 x 8 threads: >600 s spinning vs 26 s passive at load ~110). Timing only, not numerics.
export OMP_PROC_BIND=false OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
for mode in ${MODES:-cluster cluster_friction cluster_pgs cluster_posonly hertz}; do
  for np in 1 2 4 8; do
    for thr in 1 8; do
      for rep in $(seq 1 ${REPS:-3}); do
        out=$(OMP_NUM_THREADS=$thr timeout 600 mpirun --bind-to none -np $np "$B" $mode 2>&1)
        line=$(echo "$out" | grep '^MOMENTUM')
        [ -z "$line" ] && line="MOMENTUM mode=$mode np=$np thr=$thr ERROR $(echo "$out" | tail -n 2 | tr '\n' ' ')"
        echo "$line rep=$rep"
      done
    done
  done
done
echo "EXIT matrix"
