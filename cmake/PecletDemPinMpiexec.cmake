# Pin the MPI launcher to the MPI compiler's own prefix (the same rule as core's
# cmake/PecletCorePinMpiexec.cmake). FindMPI finds mpicxx deterministically but searches
# MPIEXEC_EXECUTABLE on PATH, so a foreign launcher earlier on PATH (ParaView's bundled MPICH
# mpiexec) is picked up while the binaries link the system MPI. Every rank then inits as a
# singleton: `mpiexec -n 2` runs two independent serial processes, the np >= 2 ctests "pass"
# without communicating, and the WILL_FAIL mutation controls fail (2026-09-26, a fresh dem build
# tree: mutants 1, 2, 4, 5 undetected). The launcher beside mpicxx always belongs to the same MPI.
# Scheduler-launcher clusters (srun) pass -DPECLET_DEM_PIN_MPIEXEC=OFF and set MPIEXEC_EXECUTABLE.
option(PECLET_DEM_PIN_MPIEXEC "Pin MPIEXEC_EXECUTABLE to the MPI compiler's prefix" ON)
if(PECLET_DEM_PIN_MPIEXEC AND MPI_CXX_COMPILER)
  get_filename_component(_dem_mpi_bin "${MPI_CXX_COMPILER}" DIRECTORY)
  unset(_dem_mpi_launcher CACHE)
  find_program(_dem_mpi_launcher NAMES mpirun mpiexec HINTS "${_dem_mpi_bin}" NO_DEFAULT_PATH)
  if(_dem_mpi_launcher AND NOT _dem_mpi_launcher STREQUAL "${MPIEXEC_EXECUTABLE}")
    message(STATUS "dem: MPIEXEC_EXECUTABLE was '${MPIEXEC_EXECUTABLE}' -- pinning to "
                   "'${_dem_mpi_launcher}' (matches ${MPI_CXX_COMPILER})")
    set(MPIEXEC_EXECUTABLE "${_dem_mpi_launcher}" CACHE FILEPATH "MPI launcher" FORCE)
  endif()
  unset(_dem_mpi_launcher CACHE)
endif()
