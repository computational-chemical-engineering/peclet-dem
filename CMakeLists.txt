cmake_minimum_required(VERSION 3.24)

# One version source per repo (suite/docs/QUALITY_PLAN.md D4): pyproject.toml's `version = "x.y.z"`.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _peclet_dem_version_line REGEX "^version = \"")
string(REGEX REPLACE "^version = \"([0-9]+\.[0-9]+\.[0-9]+).*$" "\\1" PECLET_DEM_VERSION
       "${_peclet_dem_version_line}")
if(NOT PECLET_DEM_VERSION MATCHES "^[0-9]+\.[0-9]+\.[0-9]+$")
  message(FATAL_ERROR "dem: could not read `version = \"x.y.z\"` from pyproject.toml")
endif()
project(dem VERSION ${PECLET_DEM_VERSION} LANGUAGES CXX)

# The `peclet.dem` module: the Kokkos + ArborX XPBD DEM engine (CUDA / HIP / OpenMP backends, selected
# by the install prefix). Kokkos and ArborX are found via find_package against the bootstrapped install
# prefix (../extern/install/<backend>, built once by ../tools/bootstrap_deps.sh -- a HARD build
# dependency); nanobind via the active Python interpreter (SuiteNanobind). With nvcc, put it on PATH
# (export PATH=/usr/local/cuda-13.2/bin:$PATH).
#
# Build (single-rank Python module):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
#   cmake --build build -j        ->  build/peclet/dem/_dem.*.so   (PYTHONPATH=build; import peclet.dem)
#
# Multi-rank: -DPECLET_DEM_MPI=ON links MPI + exposes init_mpi/enable_mpi_step/step_mpi in the module; the
# distributed step is also covered by the tests/kokkos_mpi ctests (run under mpirun).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PECLET_DEM_MPI "Build the dem module with the distributed (MPI) step exposed" OFF)
# The test suites (tests/kokkos, tests/arborx, tests/kokkos_mpi under PECLET_DEM_MPI, plus the Python
# tests under tests/python) are built from THIS tree under one option -- OFF by default so a wheel
# build is unaffected, ON in CI and in the CLAUDE.md dev recipe (`ctest --test-dir build`).
option(PECLET_DEM_BUILD_TESTS "Build the C++ test suites and register the Python tests with ctest" OFF)

# Redistributable single-GPU CUDA wheel: libcudart is provided by the `nvidia-cuda-runtime` PyPI
# dependency (installed to site-packages/nvidia/cu13/lib), not the system. ON sets the module's RPATH
# to reach that wheel from its install location so `import peclet.dem` works with no system CUDA. Used
# by the peclet-dem-cu13 packaging (packaging/pyproject-cuda.toml + the release workflow's cuda job).
# Mirrors peclet-flow.
option(PECLET_CUDA_RUNTIME_WHEEL "RPATH the module to the nvidia-cuda-runtime wheel's libcudart" OFF)

# Dependencies via the vendored PecletDeps helper: installed Kokkos+ArborX prefix + sibling checkout for
# the dev/suite build, or FetchContent-built Kokkos+ArborX + fetched core headers for a
# self-contained sdist/wheel (cibuildwheel). See cmake/PecletDeps.cmake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PecletDeps)
peclet_require_kokkos()
peclet_require_arborx()
peclet_require_nanobind()   # nanobind + core's View<->ndarray zero-copy bridge
peclet_sibling_include(peclet-core "${PECLET_CORE_TAG}" "../core" PECLET_CORE_INCLUDE)

# NB_STATIC: bundle nanobind's runtime into the module; the Kokkos device path is routed by the
# launch compiler regardless. NOMINSIZE: nanobind's default -Os is rejected by nvcc ("'s': expected a
# number") since Kokkos device sources compile as CXX through the launch compiler.
nanobind_add_module(dem NB_STATIC NOMINSIZE src/dem_bindings.cpp)
if(Kokkos_ENABLE_HIP)
  # HIP/lld experiment 1 (docs/RELEASE.md §8): nanobind sets CXX_VISIBILITY_PRESET hidden, and under hipcc
  # the host objects then reference the Kokkos SharedAllocationRecord / shared_ptr control-block vtables
  # as hidden symbols that nothing defines ("undefined hidden symbol: vtable for ..."). Default visibility
  # on the HIP path only; CUDA/OpenMP builds are untouched.
  set_target_properties(dem PROPERTIES CXX_VISIBILITY_PRESET default)
endif()
# -> peclet.dem._dem (NB_MODULE(_dem)), re-exported by peclet/dem/__init__.py. The __init__ is kept as a
# plain file under packaging/ (OUTSIDE any importable peclet/ dir, so an incomplete source package can
# never shadow the installed one) and staged into <build>/peclet/dem so `PYTHONPATH=<build> python …`
# (`import peclet.dem`) works in the dev loop too.
set_target_properties(dem PROPERTIES OUTPUT_NAME _dem
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/dem")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/dem_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/dem/__init__.py COPYONLY)
# Pure-Python particle builder (SDF -> grid + surface shell + mass properties) shipped alongside.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/particle_builder.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/dem/particle_builder.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/scene_particle.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/dem/scene_particle.py COPYONLY)
target_include_directories(dem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${PECLET_CORE_INCLUDE})
target_link_libraries(dem PRIVATE ArborX::ArborX Kokkos::kokkos)

if(PECLET_DEM_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(dem PRIVATE PECLET_DEM_MPI=1)
  target_link_libraries(dem PRIVATE MPI::MPI_CXX)
  message(STATUS "dem: distributed step ENABLED (MPI)")
endif()

# Redistributable CUDA wheel: point the module at the nvidia-cuda-runtime wheel's libcudart via a
# relative $ORIGIN RPATH (peclet/dem -> site-packages -> nvidia/cu13/lib).
if(PECLET_CUDA_RUNTIME_WHEEL)
  set_target_properties(dem PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

# --- tests -------------------------------------------------------------------------------------
if(PECLET_DEM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/kokkos)      # 8 kernel unit tests (Kokkos only)
  add_subdirectory(tests/arborx)      # broad-phase + full single-rank pipeline (2)
  if(PECLET_DEM_MPI)
    add_subdirectory(tests/kokkos_mpi)  # distributed step / migration / rebalance, np = 1, 2, 4 (24)
  endif()
  # Python tests (pytest functions under tests/python) against the module built in this tree. The
  # MPI-only files under tests/python/mpi skip themselves unless mpi4py + peclet.core.mpi are
  # importable; under PECLET_DEM_MPI they are additionally launched through mpirun at np = 1, 2, 4
  # (each returns 77 = ctest SKIP when the Python MPI stack is missing, never a silent pass).
  add_test(NAME python_tests
           COMMAND ${Python_EXECUTABLE} -m pytest -q --ignore=tests/python/mpi
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
  # The build tree is PREPENDED to PYTHONPATH at test time (the caller's PYTHONPATH -- e.g. a
  # built peclet.core python tree for the MPI tests -- is kept).
  set_tests_properties(python_tests PROPERTIES
    ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${CMAKE_CURRENT_BINARY_DIR}"
    LABELS python)
  if(PECLET_DEM_MPI)
    if(NOT MPIEXEC_EXECUTABLE)
      set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
    endif()
    if(NOT MPIEXEC_NUMPROC_FLAG)
      set(MPIEXEC_NUMPROC_FLAG -np)
    endif()
    separate_arguments(_mpi_preflags NATIVE_COMMAND "${MPIEXEC_PREFLAGS}")
    foreach(t validate_exact validate_periodic verify_distributed verify_rotating_drum_mpi)
      foreach(np 1 2 4)
        add_test(NAME python_mpi_${t}_np${np}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_mpi_preflags}
                         ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/mpi/test_${t}.py)
        set_tests_properties(python_mpi_${t}_np${np} PROPERTIES
          ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${CMAKE_CURRENT_BINARY_DIR}"
          SKIP_RETURN_CODE 77 PROCESSORS ${np} LABELS "python;mpi")
      endforeach()
    endforeach()
  endif()
endif()

# --- pip / scikit-build-core install rule -------------------------------------------------------
# `pip install .` (driven by pyproject.toml) builds against a Kokkos+ArborX prefix on
# CMAKE_PREFIX_PATH and installs the extension as `peclet.dem`. The rule is inert for a
# plain `cmake --build` (no `cmake --install`), so the developer workflow is unchanged. SKBUILD is set
# by scikit-build-core; SKBUILD_PROJECT_NAME-style guards keep it from interfering elsewhere.
if(DEFINED SKBUILD)
  install(TARGETS dem LIBRARY DESTINATION peclet/dem COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/dem_init.py
          DESTINATION peclet/dem RENAME __init__.py COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/particle_builder.py
          DESTINATION peclet/dem COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/scene_particle.py
          DESTINATION peclet/dem COMPONENT python)
endif()
