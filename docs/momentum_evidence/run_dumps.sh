#!/usr/bin/env bash
# Final-state dumps (test_momentum_mpi --dump) at OMP_NUM_THREADS=1 for the bitwise gates.
# Usage: docs/momentum_evidence/run_dumps.sh <build-dir> <out-dir> <np> <tag> <mode...>
#   writes <out-dir>/np<np>_<mode>_<tag>.bin
B=$1/tests/kokkos_mpi/test_momentum_mpi; OUT=$2; NP=$3; TAG=$4; shift 4
mkdir -p "$OUT"
export OMP_PROC_BIND=false OMP_NUM_THREADS=1
for mode in "$@"; do
  timeout 600 mpirun --bind-to none -np $NP "$B" $mode --dump="$OUT/np${NP}_${mode}_${TAG}.bin" 2>&1 \
    | grep -E '^MOMENTUM|FAILED' | sed "s/\$/ tag=$TAG/"
done
