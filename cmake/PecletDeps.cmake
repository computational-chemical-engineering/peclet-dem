# PecletDeps.cmake — self-contained dependency provisioning for a peclet compute package.
#
# A peclet compute wheel must build both ways:
#   * DEV / suite build — Kokkos (+ArborX) come from an installed prefix on CMAKE_PREFIX_PATH
#     (../extern/install/<backend> from tools/bootstrap_deps.sh) and the sibling headers
#     (core, morton) from ../<sibling>/include. Fast; the developer workflow.
#   * SELF-CONTAINED sdist/wheel (cibuildwheel) — the umbrella and siblings are ABSENT (the build runs on
#     an isolated copy of this one repo), so everything is FetchContent-built at the suite-pinned
#     versions. This is what makes `pip install peclet-flow` produce a working OpenMP CPU wheel.
#
# Selection is automatic (prefix/sibling present -> use it; else fetch), and can be forced off/on with
# -DPECLET_VENDOR_DEPS=ON. Vendored Kokkos is OpenMP+Serial only (portable CPU wheel); GPU/MPI builds use
# the prefix path. Keep PECLET_*_TAG in lockstep with tools/bootstrap_deps.sh.
include_guard(GLOBAL)
include(FetchContent)

set(PECLET_KOKKOS_TAG "5.1.1" CACHE STRING "Vendored Kokkos git tag")
set(PECLET_ARBORX_TAG "v2.1"  CACHE STRING "Vendored ArborX git tag")
set(PECLET_TPX_TAG    "v0.1.0"  CACHE STRING "Vendored core git tag (headers)")
set(PECLET_MORTON_TAG "v0.1.0"  CACHE STRING "Vendored morton git tag (headers)")
option(PECLET_VENDOR_DEPS "Force FetchContent-build of Kokkos/ArborX/siblings (self-contained wheel)" OFF)

# nanobind — found via the active interpreter (scikit-build-core supplies it as a build requirement),
# identical to the umbrella SuiteNanobind helper but vendored so an isolated sdist build needs no ../cmake.
# MUST be a macro: find_package(Python) sets variables nanobind reads at module-creation time in the
# caller's scope.
macro(peclet_require_nanobind)
  if(NOT COMMAND nanobind_add_module)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    if(NOT nanobind_DIR)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _peclet_nb OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _peclet_nb_rc)
      if(_peclet_nb_rc EQUAL 0 AND EXISTS "${_peclet_nb}")
        set(nanobind_DIR "${_peclet_nb}")
      endif()
    endif()
    find_package(nanobind CONFIG REQUIRED)
    message(STATUS "[peclet] nanobind from ${nanobind_DIR}")
  endif()
endmacro()

# Kokkos — prefix if present (unless forced), else vendored OpenMP+Serial (static, PIC).
macro(peclet_require_kokkos)
  if(NOT PECLET_VENDOR_DEPS)
    find_package(Kokkos CONFIG QUIET)
  endif()
  if(Kokkos_FOUND)
    message(STATUS "[peclet] Kokkos ${Kokkos_VERSION} from prefix (${Kokkos_DEVICES})")
  else()
    message(STATUS "[peclet] vendoring Kokkos ${PECLET_KOKKOS_TAG} (OpenMP+Serial, static PIC)")
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_Declare(kokkos GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                         GIT_TAG ${PECLET_KOKKOS_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(kokkos)
  endif()
endmacro()

# ArborX — header-only, needs Kokkos first. prefix or vendored.
macro(peclet_require_arborx)
  if(NOT PECLET_VENDOR_DEPS)
    find_package(ArborX CONFIG QUIET)
  endif()
  if(NOT ArborX_FOUND)
    message(STATUS "[peclet] vendoring ArborX ${PECLET_ARBORX_TAG}")
    FetchContent_Declare(arborx GIT_REPOSITORY https://github.com/arborx/ArborX.git
                         GIT_TAG ${PECLET_ARBORX_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(arborx)
  endif()
endmacro()

# Sibling header include dir (core / morton). Returns the sibling checkout if present, else a
# FetchContent-fetched source tree's include/ (header-only — declared but not built).
function(peclet_sibling_include repo tag sibling_reldir outvar)
  set(_local "${CMAKE_CURRENT_SOURCE_DIR}/${sibling_reldir}/include")
  if(EXISTS "${_local}" AND NOT PECLET_VENDOR_DEPS)
    set(${outvar} "${_local}" PARENT_SCOPE)
    return()
  endif()
  string(TOLOWER "peclet_sib_${repo}" _name)
  FetchContent_Declare(${_name}
    GIT_REPOSITORY "https://github.com/computational-chemical-engineering/${repo}.git"
    GIT_TAG ${tag} GIT_SHALLOW TRUE)
  FetchContent_GetProperties(${_name})
  if(NOT ${_name}_POPULATED)
    FetchContent_Populate(${_name})
  endif()
  set(${outvar} "${${_name}_SOURCE_DIR}/include" PARENT_SCOPE)
  message(STATUS "[peclet] vendored ${repo} headers -> ${${_name}_SOURCE_DIR}/include")
endfunction()
