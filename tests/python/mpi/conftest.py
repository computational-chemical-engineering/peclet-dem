"""The MPI Python tests need mpi4py + peclet.halo + a dem built with PECLET_DEM_MPI; each file
skips itself otherwise (and exits 77 = ctest SKIP when launched as a script under mpirun)."""
