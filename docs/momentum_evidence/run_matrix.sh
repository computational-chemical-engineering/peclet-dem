#!/usr/bin/env bash
# Conservation matrix: modes x np {1,2,4,8} x OMP_NUM_THREADS {1,8} x 3 repeats.
# Usage: docs/momentum_evidence/run_matrix.sh <build-dir> > matrix.txt
#        MODES="cluster_sync3 cluster_norot" docs/momentum_evidence/run_matrix.sh <build-dir>
B=${1:-build_mom}/tests/kokkos_mpi/test_momentum_mpi
export OMP_PROC_BIND=false
for mode in ${MODES:-cluster cluster_friction cluster_pgs cluster_posonly hertz}; do
  for np in 1 2 4 8; do
    for thr in 1 8; do
      for rep in 1 2 3; do
        out=$(OMP_NUM_THREADS=$thr timeout 600 mpirun --bind-to none -np $np "$B" $mode 2>&1)
        line=$(echo "$out" | grep '^MOMENTUM')
        [ -z "$line" ] && line="MOMENTUM mode=$mode np=$np thr=$thr ERROR $(echo "$out" | tail -n 2 | tr '\n' ' ')"
        echo "$line rep=$rep"
      done
    done
  done
done
echo "EXIT matrix"
