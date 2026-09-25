#!/bin/bash
# usage: wo4b_hub.sh <test_momentum_mpi> [reps8=3] : WO-4b acceptance 2-5 at np 1 -- hub_static,
# hub_ml, hub, hub_posonly, hub_pgs (step_mpi and --solo; OMP 1 once, OMP 8 x reps8) and
# cluster_periodic --solo. Prints the MOMENTUM / HUBGAP / MLCTRL / CONFLICTS / verdict lines.
B=$1; R=${2:-3}
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
for m in hub_static hub_ml hub hub_posonly hub_pgs cluster_periodic; do
  for so in "" --solo; do
    [ "$m" = cluster_periodic ] && [ -z "$so" ] && continue
    for t in 1 $(yes 8 | head -n $R); do
      echo "== $m $so OMP $t"
      OMP_NUM_THREADS=$t mpirun --bind-to none -np 1 $B $m $so 2>&1 | grep -E 'MOMENTUM|HUBGAP|MLCTRL|CONFLICTS|GATE|^OK|FAILED' | sed 's/ dL=[^ ]* dLcm=[^ ]*//'
    done
  done
done
echo EXIT
