cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")

# Auto-detect CUDA compiler in standard locations if not found in PATH
if(NOT CMAKE_CUDA_COMPILER)
    find_program(CUDA_COMPILER_SEARCH_PATH nvcc PATHS /usr/local/cuda/bin NO_DEFAULT_PATH)
    if(CUDA_COMPILER_SEARCH_PATH)
        set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_SEARCH_PATH} CACHE FILEPATH "Path to nvcc" FORCE)
        message(STATUS "Found NVCC via fallback search: ${CMAKE_CUDA_COMPILER}")
    endif()
endif()

project(demgpu LANGUAGES CXX CUDA)

# ---- Options ----
option(DEMGPU_ENABLE_MPI "Enable MPI support" ON)

# ---- C++ / CUDA Standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- External Dependencies ----

# Pybind11
find_package(pybind11 CONFIG REQUIRED)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
if(TARGET CUDA::cudart)
    message(STATUS "CUDA::cudart target exists")
else()
    message(WARNING "CUDA::cudart target missing!")
endif()

# MPI
if (DEMGPU_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

# Broad-phase: Kokkos + ArborX (replaces cuBQL; see src/cuda/broadphase.cpp). Provide via a cluster
# module or the suite's local install prefix (suite/tools/bootstrap_deps.sh) on CMAKE_PREFIX_PATH.
find_package(Kokkos CONFIG REQUIRED)
find_package(ArborX CONFIG REQUIRED)
message(STATUS "demgpu: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES}), ArborX ${ArborX_VERSION}")

# ---- Source Collection ----
# We explicitly list sources to avoid GLOB issues and follow the plan
set(DEMGPU_CUDA_SRC
    src/cuda/integration.cu
    src/cuda/solver.cu
    src/cuda/narrowphase.cu
    src/cuda/solver_velocity.cu
    src/cuda/solver_position.cu
    src/cuda/periodicity.cu
    src/cuda/periodicity.cu
    src/cuda/output_sdf.cu
    src/cuda/sdf_wrapper.cu
    src/cuda/contact_preprocessing.cu
    # Add other cuda files as they are creating
)

# on-device particle halo (CUDA-aware MPI device-pointer pack); only when transport-core is present
if (DEMGPU_ENABLE_MPI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include/tpx/halo/particle_halo.hpp")
    list(APPEND DEMGPU_CUDA_SRC src/mpi/device_particle_halo.cu)
endif()

set(DEMGPU_CPP_SRC
    src/cuda/broadphase.cpp  # ArborX broad-phase (CXX TU; Kokkos launch compiler routes to nvcc)
    src/main_binding.cpp
    src/simulation.cpp
    src/shapes/point_sampler.cpp
    src/shapes/ShapeManager.cpp
    src/cpp/OutputGenerator.cpp
    src/io/Exporter.cpp
)
# MPI-only translation units (need mpi.h); built only with MPI enabled.
if (DEMGPU_ENABLE_MPI)
    list(APPEND DEMGPU_CPP_SRC src/mpi/domain.cpp src/mpi/communicator.cpp)
endif()

# ---- Python Extension Module ----
# NO_EXTRAS: disable pybind11's LTO — it conflicts with CUDA separable compilation once Kokkos is
# linked (duplicate fatbinData / __cudaRegisterLinkedBinary at the device link).
pybind11_add_module(demgpu NO_EXTRAS ${DEMGPU_CPP_SRC} ${DEMGPU_CUDA_SRC})

# Ensure CUDA code uses separable compilation. Disable IPO/LTO: pybind11 enables it by default, but
# it conflicts with the CUDA device-link step once Kokkos is linked (duplicate `fatbinData` symbol /
# lto-wrapper failure).
set_target_properties(demgpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION OFF
)

# Windows: produce .pyd extension automatically via pybind11; ensure no 'lib' prefix
if (WIN32)
    set_target_properties(demgpu PROPERTIES PREFIX "" SUFFIX ".pyd")
endif()

# ---- Include Dirs ----
target_include_directories(demgpu PRIVATE
    src
    src/cuda
)

# ---- transport-core (shared MPI/halo library, header-only) ----
# Sibling repo providing tpx::halo::ParticleHalo / ParticleMigrator + block decomposition.
# Used by the MPI-aware step() for per-iteration owner->ghost position forwarding.
set(TPX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include")
if (DEMGPU_ENABLE_MPI AND EXISTS "${TPX_INCLUDE_DIR}/tpx/halo/particle_halo.hpp")
    target_include_directories(demgpu PRIVATE ${TPX_INCLUDE_DIR})
    target_compile_definitions(demgpu PRIVATE DEMGPU_HAVE_TPX=1)
    message(STATUS "transport-core found: ${TPX_INCLUDE_DIR} (MPI-aware step enabled)")
else()
    message(STATUS "transport-core NOT found at ${TPX_INCLUDE_DIR} (MPI-aware step disabled)")
endif()

# ---- Compile Definitions / Flags ----
if (DEMGPU_ENABLE_MPI)
    target_compile_definitions(demgpu PRIVATE DEMGPU_USE_MPI=1)
endif()


# ---- Linking ----
if (DEMGPU_ENABLE_MPI)
    target_link_libraries(demgpu PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(demgpu PRIVATE pybind11::module CUDA::cudart Kokkos::kokkos ArborX::ArborX)

# ---- Warnings ----
if (MSVC)
    target_compile_options(demgpu PRIVATE /W4)
else()
    target_compile_options(demgpu PRIVATE -Wall)
endif()

# ---- Tests ----
add_executable(test_periodicity_corner
    tests/test_periodicity_corner.cpp
    ${DEMGPU_CPP_SRC}
    ${DEMGPU_CUDA_SRC}
)
set_target_properties(test_periodicity_corner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_periodicity_corner PRIVATE src src/cuda)
target_link_libraries(test_periodicity_corner PRIVATE CUDA::cudart Kokkos::kokkos ArborX::ArborX pybind11::embed)
if (DEMGPU_ENABLE_MPI)
    target_link_libraries(test_periodicity_corner PRIVATE MPI::MPI_CXX)
    target_compile_definitions(test_periodicity_corner PRIVATE DEMGPU_USE_MPI=1)
endif()

# ---- Installation (optional) ----
install(TARGETS demgpu DESTINATION .)

message(STATUS "Configured DEMGPU with MPI=${DEMGPU_ENABLE_MPI} CUDA_ARCH=${DEMGPU_CUDA_ARCH}")
