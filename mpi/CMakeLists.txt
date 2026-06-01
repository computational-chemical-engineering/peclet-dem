# Standalone MPI validation for packing-gpu's transport-core integration (Lagrangian).
# Decoupled from the main demgpu build (no CUDA/pybind/cuBQL needed): host C++ + MPI + the header-only
# transport-core. Build:
#   cmake -S mpi -B mpi/build -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun && cmake --build mpi/build
#   ctest --test-dir mpi/build --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(demgpu_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

set(TPX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../transport-core/include")
if(NOT EXISTS "${TPX_INCLUDE}/tpx/halo/particle_migrator.hpp")
  message(FATAL_ERROR "transport-core not found at ${TPX_INCLUDE} (clone it as a sibling repo)")
endif()

enable_testing()
foreach(t test_particle_migration test_dem_step test_dem_scheme_b test_dem_scheme_c)
  add_executable(${t} ${t}.cpp)
  target_include_directories(${t} PRIVATE ${TPX_INCLUDE})
  target_link_libraries(${t} PRIVATE MPI::MPI_CXX)
endforeach()

foreach(np IN ITEMS 1 2 4)
  add_test(NAME particle_migration_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_particle_migration> ${MPIEXEC_POSTFLAGS})
  add_test(NAME dem_step_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_dem_step> ${MPIEXEC_POSTFLAGS})
  add_test(NAME dem_scheme_b_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_dem_scheme_b> ${MPIEXEC_POSTFLAGS})
  add_test(NAME dem_scheme_c_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_dem_scheme_c> ${MPIEXEC_POSTFLAGS})
endforeach()
