"""§12 S3 check: do the survey scenes (degree_survey.py: the bidisperse 'hub' beds; the RingBed-style
ring packing) still exhaust a colouring's arbitration-round cap? Runs degree_survey's scenes with its
grow() wrapped to also record sim.diagnostics.coloring_leftovers() (uncoloured manifolds, position
units) every step; prints the max over the run.

  OMP_NUM_THREADS=8 OMP_PROC_BIND=false PYTHONPATH=<build> python survey_leftovers.py bi4 bi6 ring:1500
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import degree_survey as ds  # noqa: E402

worst_left = [0, 0]


def main():
    from peclet import dem

    class Tracked(dem.Simulation):
        def step(self, *a, **k):
            r = super().step(*a, **k)
            left = self.diagnostics.coloring_leftovers()
            worst_left[0] = max(worst_left[0], left[0])
            worst_left[1] = max(worst_left[1], left[1])
            return r

    ds.dem.Simulation = Tracked
    for arg in sys.argv[1:]:
        worst_left[0] = worst_left[1] = 0
        if arg.startswith("ring"):
            steps = int(arg.split(":")[1]) if ":" in arg else 4000
            ds.run_ring(steps=steps)
        else:
            q = {"bi4": 4.0, "bi6": 6.0, "bi3": 3.0, "bi2": 2.0}[arg]
            ds.run_spheres(arg, ds.bidisperse(q), N=4000 if q >= 6 else 2000)
        print(f"LEFTOVERS case={arg} max_over_run(vel,pos)=({worst_left[0]}, {worst_left[1]})",
              flush=True)


if __name__ == "__main__":
    main()
