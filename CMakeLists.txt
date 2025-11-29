cmake_minimum_required(VERSION 3.24)
project(demgpu LANGUAGES CXX CUDA)

# ---- Options ----
option(DEMGPU_ENABLE_MPI "Enable MPI support" ON)
set(DEMGPU_CUDA_ARCH "native" CACHE STRING "CUDA architecture (e.g. 75, 86, native)")

# ---- C++ / CUDA Standards ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Pybind11 ----
find_package(pybind11 CONFIG REQUIRED)

# ---- MPI ----
if (DEMGPU_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

# ---- Source Collection ----
file(GLOB DEMGPU_CUDA_SRC CONFIGURE_DEPENDS
    src/cuda/*.cu
)
set(DEMGPU_CPP_SRC
    src/mpi/domain.cpp
    src/mpi/communicator.cpp
    src/shapes/sdf_generator.cpp
    src/shapes/point_sampler.cpp
    src/engine_binding.cpp
)

# ---- Python Extension Module ----
pybind11_add_module(demgpu ${DEMGPU_CPP_SRC} ${DEMGPU_CUDA_SRC})

# Ensure CUDA code uses separable compilation
set_target_properties(demgpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Windows: produce .pyd extension automatically via pybind11; ensure no 'lib' prefix
if (WIN32)
    set_target_properties(demgpu PROPERTIES PREFIX "" SUFFIX ".pyd")
endif()

# ---- Include Dirs ----
target_include_directories(demgpu PRIVATE src src/cuda)

# ---- Compile Definitions / Flags ----
if (DEMGPU_ENABLE_MPI)
    target_compile_definitions(demgpu PRIVATE DEMGPU_USE_MPI=1)
endif()

# Architecture flags (simple handling)
if (CMAKE_CUDA_COMPILER)
    if (DEMGPU_CUDA_ARCH STREQUAL "native")
        # Let nvcc decide or rely on default; user can override.
    else()
        target_compile_options(demgpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${DEMGPU_CUDA_ARCH},code=sm_${DEMGPU_CUDA_ARCH}>)
    endif()
endif()

# ---- Linking ----
if (DEMGPU_ENABLE_MPI)
    target_link_libraries(demgpu PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(demgpu PRIVATE pybind11::module)

# ---- Warnings ----
if (MSVC)
    target_compile_options(demgpu PRIVATE /W4)
else()
    target_compile_options(demgpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Installation (optional) ----
install(TARGETS demgpu DESTINATION .)

message(STATUS "Configured DEMGPU with MPI=${DEMGPU_ENABLE_MPI} CUDA_ARCH=${DEMGPU_CUDA_ARCH}")
