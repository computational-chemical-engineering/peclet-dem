"""peclet.dem — Lagrangian Discrete Element Method (XPBD) particle packing.

A Kokkos + ArborX XPBD solver with SDF point-shell collision for dense particle packing. The compiled
backend (Serial / OpenMP / CUDA / HIP) is chosen at build time — ``peclet.dem.execution_space`` reports
which one this build has. The distributed (MPI) step is exposed only in an MPI-enabled build
(``pip install . --config-settings=cmake.define.PECLET_DEM_MPI=ON``).

* :class:`peclet.dem.Simulation` — the packing simulation (initialize_shape, set_positions, step, ...).
* :func:`peclet.dem.build_particle` — build a general particle (grid SDF + surface point shell + mass
  properties) from an implicit-solid SDF, ready for ``Simulation.set_sdf_shape``.

``peclet`` is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages, so it has no
top-level ``__init__.py``.
"""

from ._dem import *  # noqa: F401,F403

# The particle builder is pure Python (NumPy + scikit-image); import lazily-tolerant so the compiled
# module still imports if scikit-image is absent (only build_particle needs it).
from .particle_builder import ParticleShape, build_particle  # noqa: F401

__version__ = "0.3.0"
