#!/bin/bash
# usage: dumps.sh <test_momentum_mpi binary> <outdir> : np1 OMP1 dumps of every WO-0 dump mode (+ np4 cluster, cluster_pgs)
B=$1; O=$2; mkdir -p $O
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
for m in cluster cluster_pgs cluster_posonly hertz cluster_poisson cluster_multilevel cluster_escalate \
         cluster_ordered cluster_onesided cluster_e09 cluster_e10 cluster_friction cluster_jacobi cluster_periodic \
         cluster_sync3 cluster_norot hub hub_posonly hub_pgs ring_mini tri tri_pgs; do
  mpirun --bind-to none -np 1 $B $m --dump=$O/$m.np1.omp1.dump > $O/$m.np1.omp1.log 2>&1
done
for m in cluster_periodic hub hub_posonly hub_pgs ring_mini; do
  mpirun --bind-to none -np 1 $B $m --solo --dump=$O/${m}_solo.np1.omp1.dump > $O/${m}_solo.np1.omp1.log 2>&1
done
for m in cluster cluster_pgs; do
  mpirun --bind-to none -np 4 $B $m --dump=$O/$m.np4.omp1.dump > $O/$m.np4.omp1.log 2>&1
done
echo EXIT
