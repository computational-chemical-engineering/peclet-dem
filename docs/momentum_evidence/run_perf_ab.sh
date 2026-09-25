#!/usr/bin/env bash
# Interleaved before/after timing on the run_perf.sh protocol (N = 19683, fully periodic, np 4 x 4
# and np 8 x 2 threads on cores 8-23, each rank taskset-pinned, 10 warm-up + 50 timed steps):
# every repeat runs the BEFORE build and then the AFTER build back to back, so both see the same
# host load. Usage: docs/momentum_evidence/run_perf_ab.sh <before-build> <after-build> > perf_ab.txt
D=$(cd "$(dirname "$0")" && pwd)
A=$1/tests/kokkos_mpi/test_momentum_mpi
B=$2/tests/kokkos_mpi/test_momentum_mpi
export OMP_PROC_BIND=false PIN_BASE=8
for mode in perf_gas perf_pgs; do
  for cfg in "4 4" "8 2"; do
    set -- $cfg
    for rep in 1 2 3 4 5; do
      for tag in before after; do
        bin=$A; [ $tag = after ] && bin=$B
        echo "LOAD $(uptime | sed 's/.*average: //')"
        OMP_NUM_THREADS=$2 mpirun --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x PIN_BASE \
          ${OMP_WAIT_POLICY:+-x OMP_WAIT_POLICY} \
          -np $1 "$D/pin_rank.sh" "$bin" $mode 2>&1 | grep '^PERF' | sed "s/\$/ rep=$rep build=$tag/"
      done
    done
  done
done
echo "EXIT perf"
