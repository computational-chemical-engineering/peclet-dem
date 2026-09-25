#!/usr/bin/env bash
# Defect 1 (colour overflow) matrix: the hub scenes, single-rank demStep (--solo) and step_mpi
# np {1,2,4}, OMP_NUM_THREADS {1,8}, REPS repeats. One HUB + one MOMENTUM line per run.
# Usage: docs/contact_evidence/run_hub.sh <build-dir> > docs/contact_evidence/hub.txt
B=${1:-build_ct}/tests/kokkos_mpi/test_momentum_mpi
export OMP_PROC_BIND=false OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
for mode in ${MODES:-hub hub_posonly}; do
  for cfg in solo 1 2 4; do
    for thr in 1 8; do
      for rep in $(seq 1 ${REPS:-3}); do
        if [ "$cfg" = solo ]; then
          out=$(OMP_NUM_THREADS=$thr timeout 600 "$B" $mode --solo 2>&1)
        else
          out=$(OMP_NUM_THREADS=$thr timeout 600 mpirun --bind-to none -np $cfg "$B" $mode 2>&1)
        fi
        echo "$out" | grep -E '^(HUB|MOMENTUM)' | sed "s/\$/ rep=$rep/"
        echo "$out" | grep -q '^MOMENTUM' || echo "ERROR mode=$mode cfg=$cfg thr=$thr $(echo "$out" | tail -n 2 | tr '\n' ' ')"
      done
    done
  done
done
echo "EXIT hub"
