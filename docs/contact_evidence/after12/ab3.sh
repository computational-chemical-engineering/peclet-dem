#!/usr/bin/env bash
D=/home/frankp/Codes/suite/dem-contacts/docs/momentum_evidence
declare -A BIN=([base]=/home/frankp/Codes/suite/dem-perfbase/build_pb/tests/kokkos_mpi/test_momentum_mpi [main]=/home/frankp/Codes/suite/dem-contacts/build_ct/tests/kokkos_mpi/test_momentum_mpi [new]=/home/frankp/Codes/suite/dem-perf/build_perf/tests/kokkos_mpi/test_momentum_mpi)
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive PIN_BASE=0
for G in 27 54; do for mode in perf_gas perf_pgs; do for cfg in "4 2" "8 2"; do
  set -- $cfg
  for rep in 1 2 3; do for tag in base main new; do
    PERF_G=$G OMP_NUM_THREADS=$2 mpirun --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_WAIT_POLICY -x PIN_BASE -x PERF_G \
      -np $1 $D/pin_rank.sh ${BIN[$tag]} $mode --perf-g=$G 2>&1 | grep '^PERF ' | sed "s/\$/ G=$G build=$tag load=$(cut -d' ' -f1 /proc/loadavg)/"
  done; done
done; done; done
echo EXIT
