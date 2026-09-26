#!/usr/bin/env bash
# usage: pm.sh <build> <mode> <g> <np> <thr> <out>
S=/tmp/claude-1003/-home-frankp-Codes-suite/53a85a4c-bcc1-4b4e-8987-b25c5c7a74d8/scratchpad
D=/home/frankp/Codes/suite/dem-contacts/docs/momentum_evidence
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive PIN_BASE=${PIN_BASE:-0} OMP_NUM_THREADS=$5 PERF_G=$3
export LD_PRELOAD=$S/pmpi/libpmpi.so
mpirun --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_WAIT_POLICY -x PIN_BASE -x PERF_G -x LD_PRELOAD \
  -np $4 $D/pin_rank.sh $1/tests/kokkos_mpi/test_momentum_mpi $2 --perf-g=$3 > $6 2>&1
