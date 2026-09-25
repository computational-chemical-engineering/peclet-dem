#!/usr/bin/env bash
# Defect 4 (pairs no rank sees): the three report-only ownership modes at np 1, 2, 4, 8.
# Usage: docs/contact_evidence/run_missed.sh <build-dir> > docs/contact_evidence/missed.txt
B=${1:-build_ct}/tests/kokkos_mpi/test_ownership_mpi
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive OMP_NUM_THREADS=1
for m in missed_drift_pair missed_drift_lattice missed_periodic; do
  for np in 1 2 4 8; do
    echo "### $m np=$np"
    timeout 900 mpirun --bind-to none -np $np "$B" $m 2>&1
  done
done
echo "EXIT missed"
