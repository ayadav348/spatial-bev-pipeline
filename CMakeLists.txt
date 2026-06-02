cmake_minimum_required(VERSION 3.16)
project(SpatialBEVPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Locate System Packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

# 2. Explicitly map include paths for both matrix ecosystems
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/eigen3
)

# 3. Target Executable Configuration
add_executable(bev_warp src/main.cpp)
target_link_libraries(bev_warp PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
