#!/bin/bash
# usage: wo5_matrix_cuda.sh <CUDA test_momentum_mpi> [reps=3] : WO-5 acceptance 1 on CUDA -- np 1,
# step_mpi and --solo, `reps` runs each, the projection modes of the G1 matrix.
B=$1; R=${2:-3}
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive OMP_NUM_THREADS=1
for m in cluster_pgs cluster_poisson cluster_multilevel cluster_escalate cluster_ordered cluster_posonly \
         hub_pgs hub_posonly hub_static hub_ml ring_mini; do
  for so in "" --solo; do
    for r in $(seq 1 $R); do
      timeout 900 mpirun --bind-to none -np 1 $B $m $so 2>&1 |
        grep -E '^MOMENTUM|^HUBGAP|^MLCTRL|^ORPHAN|^CONFLICTS|^GATE|^OK|^FAILED' |
        sed 's/ degVel.*//' | tr '\n' ' '
      echo " run=$r ${so:-step_mpi}"
    done
  done
done
echo EXIT
