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
add_executable(test_particle_migration test_particle_migration.cpp)
target_include_directories(test_particle_migration PRIVATE ${TPX_INCLUDE})
target_link_libraries(test_particle_migration PRIVATE MPI::MPI_CXX)

foreach(np IN ITEMS 1 2 4)
  add_test(NAME particle_migration_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_particle_migration> ${MPIEXEC_POSTFLAGS})
endforeach()
