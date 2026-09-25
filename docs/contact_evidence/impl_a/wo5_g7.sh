#!/bin/bash
# usage: wo5_g7.sh <test_momentum_mpi> <outdir> : WO-5 acceptance 6 (§13.6, OMP 1) --
#   G7a tri_pgs --vel-iters 4..64, np 1 and 2 (KE lines);
#   G7c cluster_pgs --steps=1 --vel-iters 8/32/128, np 1/2/4/8 (dumps; wo5_g7.py takes the RMS);
#   G7e cluster, cluster_pgs, cluster_friction, np 1/2/4/8 (MOMENTUM ovl);
#   G7f cluster_posonly --steps=1 --pos-iters=400, cluster_pgs --steps=1 --vel-iters=400 (ITERS).
B=$1; O=$2; mkdir -p $O
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
run() { timeout 900 mpirun --bind-to none -np "$@" 2>&1 | grep -E '^MOMENTUM|^KE|^ITERS|^GATE|^OK|^FAILED'; }
for it in 4 8 16 32 64; do for np in 1 2; do echo "== G7a it=$it np=$np"; run $np $B tri_pgs --vel-iters=$it; done; done
for it in 8 32 128; do for np in 1 2 4 8; do echo "== G7c it=$it np=$np"; run $np $B cluster_pgs --steps=1 --vel-iters=$it --dump=$O/g7c_it${it}_np${np}.dump; done; done
for m in cluster cluster_pgs cluster_friction; do for np in 1 2 4 8; do echo "== G7e $m np=$np"; run $np $B $m; done; done
for np in 1 2 4 8; do echo "== G7f pos np=$np"; run $np $B cluster_posonly --steps=1 --pos-iters=400
                      echo "== G7f vel np=$np"; run $np $B cluster_pgs --steps=1 --vel-iters=400; done
echo EXIT
