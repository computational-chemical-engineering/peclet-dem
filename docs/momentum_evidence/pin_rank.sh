#!/usr/bin/env bash
# Pin each MPI rank to its own contiguous block of OMP_NUM_THREADS physical cores starting at
# PIN_BASE (logical CPUs 0-23 are the 24 physical cores of the host; 24-47 their SMT siblings).
r=${OMPI_COMM_WORLD_RANK:?}
T=${OMP_NUM_THREADS:?}
first=$(( ${PIN_BASE:-8} + r * T ))
exec taskset -c "${first}-$(( first + T - 1 ))" "$@"
