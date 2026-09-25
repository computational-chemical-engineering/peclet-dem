#!/usr/bin/env bash
# Defect 2 (legacy-friction couple): the one-pair analytic check and the cluster_friction floor
# versus dt and position iterations. np = 1, OMP_NUM_THREADS = 1 (deterministic).
# Usage: docs/contact_evidence/run_friction.sh <build-dir> > docs/contact_evidence/friction.txt
B=${1:-build_ct}/tests/kokkos_mpi/test_momentum_mpi
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive OMP_NUM_THREADS=1
for m in friction_pair friction_pair_pgs; do
  for d in 0.001 0.01 0.05 0.1 0.2; do "$B" $m --delta=$d | grep '^FRICPAIR'; done
done
for dt in 0.02 0.005 0.0025; do "$B" friction_pair --delta=0.05 --dt=$dt | grep '^FRICPAIR'; done
for posit in 20 5 2; do
  for dt in 0.02 0.01 0.005 0.0025; do
    out=$("$B" cluster_friction --dt=$dt --posit=$posit 2>&1)
    echo "$out" | grep -E '^(MOMENTUM|FRIC )' | tr '\n' ' '; echo "posit=$posit"
  done
done
for dt in 0.01 0.0025; do
  out=$("$B" cluster_pgs --dt=$dt 2>&1); echo "$out" | grep '^MOMENTUM'
done
echo "EXIT friction"
