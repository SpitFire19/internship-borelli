cmake_minimum_required(VERSION 3.16)

project(DelaunaySlidingWindow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Gudhi location
set(GUDHI_DIR "$ENV{HOME}/gudhi-devel")


# ============================================================
# Shared project code
# ============================================================

add_library(delaunay_core
    src/geometry.cpp
    src/algorithms.cpp
    src/debug.cpp
)

target_include_directories(delaunay_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GUDHI_DIR}/src/Simplex_tree/include"
    "${GUDHI_DIR}/src/Alpha_complex/include"
    "${GUDHI_DIR}/src/common/include"
    "/usr/include/eigen3"
)

target_link_libraries(delaunay_core PUBLIC
    gmp
    mpfr
    csv
)


# ============================================================
# Sliding window benchmark
# ============================================================

add_executable(sliding_window
    src/delaunay_sliding_window_benchmark.cpp
)

target_link_libraries(sliding_window PRIVATE
    delaunay_core
)


# ============================================================
# Full rebuild benchmark
# ============================================================

add_executable(recalculate
    src/complex_recalculate_benchmark.cpp
)

target_link_libraries(recalculate PRIVATE
    delaunay_core
)
