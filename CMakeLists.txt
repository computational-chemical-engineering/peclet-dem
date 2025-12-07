cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
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

# cuBQL
include(FetchContent)
FetchContent_Declare(
  cuBQL
  GIT_REPOSITORY https://github.com/NVIDIA/cuBQL.git
  GIT_TAG        main 
)
FetchContent_MakeAvailable(cuBQL)
message(STATUS "cuBQL source dir: ${cuBQL_SOURCE_DIR}")
message(STATUS "cubql source dir: ${cubql_SOURCE_DIR}")

# ---- Source Collection ----
# We explicitly list sources to avoid GLOB issues and follow the plan
set(DEMGPU_CUDA_SRC
    src/cuda/integration.cu
    src/cuda/broadphase.cu
    src/cuda/solver.cu 
    # Add other cuda files as they are created
)

set(DEMGPU_CPP_SRC
    src/main_binding.cpp
    src/simulation.cpp
    src/mpi/domain.cpp
    src/mpi/communicator.cpp
    src/shapes/point_sampler.cpp
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
target_include_directories(demgpu PRIVATE 
    src 
    src/cuda
    ${cuBQL_SOURCE_DIR} # Include cuBQL headers
)

# ---- Compile Definitions / Flags ----
if (DEMGPU_ENABLE_MPI)
    target_compile_definitions(demgpu PRIVATE DEMGPU_USE_MPI=1)
endif()


# ---- Linking ----
if (DEMGPU_ENABLE_MPI)
    target_link_libraries(demgpu PRIVATE MPI::MPI_CXX)
endif()

target_link_libraries(demgpu PRIVATE pybind11::module CUDA::cudart cuBQL_queries cuBQL_cuda_float3_static)

# ---- Warnings ----
if (MSVC)
    target_compile_options(demgpu PRIVATE /W4)
else()
    target_compile_options(demgpu PRIVATE -Wall)
endif()

# ---- Installation (optional) ----
install(TARGETS demgpu DESTINATION .)

message(STATUS "Configured DEMGPU with MPI=${DEMGPU_ENABLE_MPI} CUDA_ARCH=${DEMGPU_CUDA_ARCH}")
