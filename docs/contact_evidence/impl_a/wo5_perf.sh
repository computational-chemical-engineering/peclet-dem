#!/bin/bash
# usage: wo5_perf.sh <base test_momentum_mpi> <after test_momentum_mpi> [reps=5] : WO-5
# acceptance 9 / G12 -- perf_pgs and perf_gas at np x OMP = 8 x 2 and 4 x 4, `reps` interleaved
# repeats (base, after, base, after, ...), the 1-min load recorded before each pair. Prints the
# PERF lines; the medians and ratios: wo5_perf.sh ... | python3 - <<< (see IMPL_A WO-5).
A=$1; B=$2; R=${3:-5}
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
for cfg in "8 2" "4 4"; do
  set -- $cfg; np=$1; t=$2
  for mode in perf_pgs perf_gas; do
    for r in $(seq 1 $R); do
      echo "LOAD $(cut -d' ' -f1 /proc/loadavg) mode=$mode np=$np thr=$t rep=$r"
      echo -n "base  "; OMP_NUM_THREADS=$t mpirun --bind-to none -np $np $A $mode 2>&1 | grep '^PERF'
      echo -n "after "; OMP_NUM_THREADS=$t mpirun --bind-to none -np $np $B $mode 2>&1 | grep '^PERF'
    done
  done
done
echo EXIT
