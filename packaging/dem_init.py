"""peclet.dem — Lagrangian Discrete Element Method (XPBD) particle packing.

A Kokkos + ArborX XPBD solver with SDF point-shell collision for dense particle packing. The compiled
backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.dem.execution_space`` reports
which one this build has. The distributed (MPI) step is exposed only in an MPI-enabled build
(``pip install . --config-settings=cmake.define.PECLET_DEM_MPI=ON``).

* :class:`peclet.dem.Simulation` — the packing simulation (initialize_shape, set_positions, step, ...).

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._dem import *  # noqa: F401,F403

__version__ = "0.1.0"
