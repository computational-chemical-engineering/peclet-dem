#!/bin/bash
# usage: inert_test.sh <binA> <binB> <outdir>  -- existing test_momentum_mpi modes, np1 (+np4 cluster/cluster_pgs), OMP 1
A=$1; B=$2; O=$3; mkdir -p $O
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive
run() { # tag np bin args...
  local tag=$1 np=$2 bin=$3; shift 3
  mpirun --bind-to none -np $np $bin "$@" --dump=$O/$tag.dump > $O/$tag.log 2>&1
}
cases=(
"cluster 1" "cluster_friction 1" "cluster_pgs 1" "cluster_posonly 1" "hertz 1" "cluster_sync3 1"
"cluster_norot 1" "cluster_periodic 1" "cluster_jacobi 1" "hub 1" "hub_posonly 1"
"cluster 4" "cluster_pgs 4")
for c in "${cases[@]}"; do set -- $c; m=$1; np=$2
  run A_${m}_np$np $np $A $m; run B_${m}_np$np $np $B $m
done
for m in cluster_periodic hub; do run A_${m}_solo 1 $A $m --solo; run B_${m}_solo 1 $B $m --solo; done
for m in friction_pair friction_pair_pgs; do
  mpirun --bind-to none -np 1 $A $m > $O/A_$m.log 2>&1; mpirun --bind-to none -np 1 $B $m > $O/B_$m.log 2>&1; done
nd=0; nl=0; tot=0
for f in $O/A_*.log; do t=$(basename $f .log); t=${t#A_}; tot=$((tot+1))
  if [ -f $O/A_$t.dump ]; then cmp -s $O/A_$t.dump $O/B_$t.dump && echo "dump $t IDENT" || { echo "dump $t DIFF"; nd=$((nd+1)); }; fi
  a=$(grep -E "^(MOMENTUM|HUB|FRIC|FRICPAIR) " $O/A_$t.log | sort); b=$(grep -E "^(MOMENTUM|HUB|FRIC|FRICPAIR) " $O/B_$t.log | sort)
  [ "$a" == "$b" ] && [ -n "$a" ] && echo "lines $t IDENT" || { echo "lines $t DIFF"; nl=$((nl+1)); }
done
echo "EXIT cases=$tot dumpdiff=$nd linediff=$nl"
