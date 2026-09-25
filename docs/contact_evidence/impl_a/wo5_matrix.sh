#!/bin/bash
# usage: wo5_matrix.sh <test_momentum_mpi> [nps="1 2 4 8"] [reps8=3] : WO-5 acceptance 1-3, 7 --
# the G1 conservation matrix of the projection modes (§13.5 WO-5): OMP 1 once, OMP 8 x reps8, per
# np. One line per run: MOMENTUM (+ HUBGAP / MLCTRL / ORPHAN / CONFLICTS pos/vel) + verdict.
B=$1; NPS=${2:-"1 2 4 8"}; R=${3:-3}
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
for m in cluster_pgs cluster_poisson cluster_multilevel cluster_escalate cluster_ordered cluster_posonly \
         hub_pgs hub_posonly hub_static hub_ml ring_mini; do
  for np in $NPS; do
    for t in 1 $(yes 8 | head -n $R); do
      OMP_NUM_THREADS=$t timeout 900 mpirun --bind-to none -np $np $B $m 2>&1 |
        grep -E '^MOMENTUM|^HUBGAP|^MLCTRL|^ORPHAN|^CONFLICTS|^GATE|^OK|^FAILED' |
        sed 's/ degVel.*//' | tr '\n' ' '
      echo
    done
  done
done
echo EXIT
