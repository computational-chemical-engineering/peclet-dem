"""Shared import guard for the MPI Python tests.

Import as `from _mpi_common import *`-free explicit names: SKIP is None when the whole MPI Python
stack is available (mpi4py, peclet.core.mpi, a dem built with -DPECLET_DEM_MPI=ON), else the reason.
Running a test file as a script (`mpirun -np N python test_x.py`) exits 77 -- the ctest SKIP return
code registered by the root CMake -- instead of pretending to pass.
"""
import sys

SKIP = None
try:
    from mpi4py import MPI  # noqa: F401
    from peclet import dem
    from peclet.core import mpi as core_mpi  # noqa: F401
    if not hasattr(dem.Simulation, "init_mpi"):
        SKIP = "peclet.dem was built without PECLET_DEM_MPI (no init_mpi)"
except ImportError as exc:
    SKIP = f"MPI Python stack unavailable: {exc}"

try:
    import pytest
    skip_unless_mpi = pytest.mark.skipif(SKIP is not None, reason=SKIP or "")
except ImportError:  # script mode without pytest installed
    def skip_unless_mpi(fn):
        return fn


def script_main(*tests):
    """Run the given test functions as an mpirun script; exit 77 when the stack is missing."""
    if SKIP is not None:
        print(f"SKIP: {SKIP}")
        sys.exit(77)
    for t in tests:
        t()
    sys.exit(0)
