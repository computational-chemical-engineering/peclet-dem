#!/bin/bash
# usage: wo4b_refdumps.sh <build dir> <outdir> : np 1 OMP 1 dumps of every WO-0 dump mode (+ np 4
# cluster, cluster_pgs; --solo variants), the 5 S6 wall scenes and the 150-step multilevel pile.
# Compare two outdirs with cmp (*.dump) and impl_a/cmpnpz.py (*.npz).
B=$(realpath $1); O=$2; mkdir -p $O/s6 $O/pile
H=$(dirname $(realpath $0))
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
$H/wo0_dumps.sh $B/tests/kokkos_mpi/test_momentum_mpi $O > /dev/null
PYTHONPATH=$B python $H/s6_wall_dumps.py dump $O/s6 > $O/s6.log 2>&1
PYTHONPATH=$B python $H/mlpile.py $O/pile 150 > $O/pile.log 2>&1
echo EXIT
