cmake_minimum_required(VERSION 3.16)
project(SpatialBEVPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Locate System Packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

# 2. Target Executable Configuration
add_executable(bev_warp
    src/main.cpp
    src/kalman_filter.cpp
)

# 3. Modern Target-Based Include Directories
target_include_directories(bev_warp PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 4. Link Libraries & Dependencies
target_link_libraries(bev_warp PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# 5. Unit Tests (Kalman Filter) — no external framework, runs under CTest
enable_testing()

add_executable(test_kalman
    tests/test_kalman.cpp
    src/kalman_filter.cpp
)
target_include_directories(test_kalman PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_kalman PRIVATE Eigen3::Eigen)

add_test(NAME kalman_unit_tests COMMAND test_kalman)
