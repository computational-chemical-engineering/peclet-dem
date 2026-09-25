#!/bin/bash
# usage: wo5_dumps.sh <build dir> <outdir> : WO-5 G3 -- np 1 OMP 1 final-state dumps of every
# test_momentum_mpi mode under step_mpi AND --solo (hertz: step_mpi only), the 5 S6 wall scenes
# and the 150-step multilevel pile. Compare two outdirs with cmp (*.dump) and cmpnpz.py (*.npz).
B=$(realpath $1); O=$2; mkdir -p $O/s6 $O/pile
H=$(dirname $(realpath $0))
T=$B/tests/kokkos_mpi/test_momentum_mpi
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
for m in cluster cluster_pgs cluster_posonly hertz cluster_poisson cluster_multilevel cluster_escalate \
         cluster_ordered cluster_onesided cluster_e09 cluster_e10 cluster_friction cluster_jacobi \
         cluster_periodic cluster_sync3 cluster_norot hub hub_posonly hub_pgs hub_static hub_ml \
         ring_mini tri tri_pgs; do
  mpirun --bind-to none -np 1 $T $m --dump=$O/$m.np1.dump > $O/$m.np1.log 2>&1
  [ $m = hertz ] && continue
  mpirun --bind-to none -np 1 $T $m --solo --dump=$O/${m}_solo.np1.dump > $O/${m}_solo.np1.log 2>&1
done
PYTHONPATH=$B python $H/s6_wall_dumps.py dump $O/s6 > $O/s6.log 2>&1
PYTHONPATH=$B python $H/mlpile.py $O/pile 150 > $O/pile.log 2>&1
echo EXIT
