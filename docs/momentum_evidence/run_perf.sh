#!/usr/bin/env bash
# "Before" performance: distributed XPBD step, N = 19683 (27^3), fully periodic, 16 physical cores
# (np 4 x 4 threads, np 8 x 2 threads; cores 8-23), each rank taskset-pinned; 5 runs each, 10
# warm-up steps excluded, 50 timed. Records `uptime` before/after each run.
# Usage: docs/momentum_evidence/run_perf.sh <build-dir> > perf.txt
D=$(cd "$(dirname "$0")" && pwd)
B=${1:-build_mom}/tests/kokkos_mpi/test_momentum_mpi
export OMP_PROC_BIND=false PIN_BASE=8
for mode in perf_gas perf_pgs; do
  for cfg in "4 4" "8 2"; do
    set -- $cfg
    for rep in 1 2 3 4 5; do
      echo "LOAD before $(uptime | sed 's/.*average: //')"
      OMP_NUM_THREADS=$2 mpirun --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x PIN_BASE \
        -np $1 "$D/pin_rank.sh" "$B" $mode 2>&1 | grep '^PERF' | sed "s/\$/ rep=$rep/"
      echo "LOAD after  $(uptime | sed 's/.*average: //')"
    done
  done
done
echo "EXIT perf"
