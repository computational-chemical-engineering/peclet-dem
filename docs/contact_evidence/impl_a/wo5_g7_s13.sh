#!/bin/bash
# usage: wo5_g7_s13.sh <test_momentum_mpi> <outdir> : G7 as restated by §12 S13 / S18 (OMP 1) --
#   G7a tri_pgs --axis=2 --no-stop --vel-iters 4..64, np 1 and 2 (KE lines; axis 2 crosses the
#       np 2 face, S13: every adaptive stop off, N forced);
#   G7c cluster_pgs --steps=1 --no-stop --vel-iters 8/32/128/256, np 1/2/4/8 (dumps);
#   G7e cluster, cluster_pgs, cluster_friction, np 1/2/4/8, stops ON (MOMENTUM ovl);
#   G7f cluster_posonly --steps=1 --pos-iters=400, cluster_pgs --steps=1 --vel-iters=400, stops
#       ON with S14's consensus residual (ITERS).
# Tables: python wo5_g7_s13.py <this output> <outdir>.
B=$1; O=$2; mkdir -p $O
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
run() { timeout 900 mpirun --bind-to none -np "$@" 2>&1 | grep -E '^MOMENTUM|^KE|^ITERS|^GATE|^OK|^FAILED'; }
for it in 4 8 16 32 64; do for np in 1 2; do echo "== G7a it=$it np=$np"; run $np $B tri_pgs --axis=2 --no-stop --vel-iters=$it; done; done
for it in 8 32 128 256; do for np in 1 2 4 8; do echo "== G7c it=$it np=$np"; run $np $B cluster_pgs --steps=1 --no-stop --vel-iters=$it --dump=$O/g7c_it${it}_np${np}.dump; done; done
for m in cluster cluster_pgs cluster_friction; do for np in 1 2 4 8; do echo "== G7e $m np=$np"; run $np $B $m; done; done
for np in 1 2 4 8; do echo "== G7f pos np=$np"; run $np $B cluster_posonly --steps=1 --pos-iters=400
                      echo "== G7f vel np=$np"; run $np $B cluster_pgs --steps=1 --vel-iters=400; done
echo EXIT
