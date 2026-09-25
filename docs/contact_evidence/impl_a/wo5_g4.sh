#!/bin/bash
# usage: wo5_g4.sh <test_momentum_mpi> <outdir> [runs=5] : WO-5 acceptance 5 (G4) -- np 4 and 8,
# OMP 1, `runs` dumps per mode; prints IDENT / DIFF per (mode, np) against run 1.
B=$1; O=$2; N=${3:-5}; mkdir -p $O
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
for m in cluster_pgs cluster_poisson cluster_posonly hub_pgs hub_ml ring_mini; do
  for np in 4 8; do
    for r in $(seq 1 $N); do
      timeout 900 mpirun --bind-to none -np $np $B $m --dump=$O/$m.np$np.r$r.dump > /dev/null 2>&1
    done
    ok=IDENT
    for r in $(seq 2 $N); do cmp -s $O/$m.np$np.r1.dump $O/$m.np$np.r$r.dump || ok=DIFF; done
    echo "$ok $m np=$np runs=$N"
  done
done
echo EXIT
