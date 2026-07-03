cmake_minimum_required(VERSION 3.16)

project(DelaunaySlidingWindow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gudhi location
set(GUDHI_DIR "$ENV{HOME}/gudhi-devel")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GUDHI_DIR}/src/Simplex_tree/include
    ${GUDHI_DIR}/src/Alpha_complex/include
    ${GUDHI_DIR}/src/common/include
    /usr/include/eigen3
)

# Sliding window benchmark
add_executable(sliding_window
    delaunay_sliding_window_benchmark.cpp
)

target_link_libraries(sliding_window
    gmp
    mpfr
    csv
)

# Full rebuild benchmark
add_executable(recalculate
    complex_recalculate_benchmark.cpp
)

target_link_libraries(recalculate
    gmp
    mpfr
    csv
)