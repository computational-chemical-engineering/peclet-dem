#!/bin/bash
# WO-0 baselines (docs/contact_solve_framework.md §8 WO-0 item 4), from the frozen build_base.
# usage: docs/contact_evidence/baseline/run_baseline.sh [build_dir=build_base] [outdir=docs/contact_evidence/baseline]
#   host part: np 1 OMP 1 --dump references, the report-line matrix (np 1,2,4,8 x OMP 1,8) and
#   the oracle at np 1,2,4,8. CUDA part: run_baseline.sh cuda [build_ct_cuda] [outdir].
set -u
export OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
if [ "${1:-}" == "cuda" ]; then
  B=${2:-build_ct_cuda}/tests/kokkos_mpi/test_momentum_mpi; O=${3:-docs/contact_evidence/baseline}
  export OMP_NUM_THREADS=1
  f=$O/cuda_lines.txt; : > $f
  for m in ring_mini hub hub_posonly; do for solo in "" "--solo"; do for rep in 1 2 3; do
    echo "## $m ${solo:-step_mpi} np=1 run=$rep" >> $f
    mpirun --bind-to none -np 1 $B $m $solo 2>&1 | grep -E "^(MOMENTUM|KE|CONFLICTS|HUB|OK|FAILED)" >> $f
  done; done; done
  echo EXIT; exit 0
fi
BD=${1:-build_base}; O=${2:-docs/contact_evidence/baseline}; mkdir -p $O
M=$BD/tests/kokkos_mpi/test_momentum_mpi; W=$BD/tests/kokkos_mpi/test_ownership_mpi
# 1. byte-identity references, np 1 OMP 1
export OMP_NUM_THREADS=1
for m in cluster cluster_pgs cluster_posonly hertz cluster_poisson cluster_multilevel cluster_escalate \
         cluster_ordered cluster_onesided cluster_e09 cluster_e10 cluster_friction cluster_jacobi \
         cluster_periodic hub hub_posonly hub_pgs ring_mini tri tri_pgs; do
  mpirun --bind-to none -np 1 $M $m --dump=$O/$m.np1.omp1.dump > /dev/null 2>&1
done
for m in cluster_periodic hub hub_posonly hub_pgs ring_mini; do
  mpirun --bind-to none -np 1 $M $m --solo --dump=$O/${m}_solo.np1.omp1.dump > /dev/null 2>&1
done
# 2. report lines of every new mode (+ hub, hub_posonly), np 1,2,4,8 x OMP 1,8
f=$O/lines.txt; : > $f
for m in tri tri_pgs cluster_e09 cluster_e10 cluster_poisson cluster_multilevel cluster_escalate \
         cluster_ordered cluster_onesided hub hub_posonly hub_pgs ring_mini; do
  for np in 1 2 4 8; do for t in 1 8; do
    echo "## $m np=$np OMP=$t" >> $f
    OMP_NUM_THREADS=$t mpirun --bind-to none -np $np $M $m 2>&1 | grep -E "^(MOMENTUM|KE|CONFLICTS|HUB|OK|FAILED)" >> $f
  done; done
done
# 3. the oracle, np 1,2,4,8 (OMP 1)
f=$O/oracle.txt; : > $f
for np in 1 2 4 8; do for m in oracle_closed oracle_shear oracle_periodic; do
  mpirun --bind-to none -np $np $W $m 2>&1 | grep -E "^(ORACLE|  first|OK|FAILED)" >> $f
done; done
echo EXIT
