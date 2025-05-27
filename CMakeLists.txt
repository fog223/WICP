cmake_minimum_required(VERSION 3.26)
project(WICP)

set(CMAKE_CXX_STANDARD 17)

# Include
include_directories(./)
find_package(PCL CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Some fixes for the Glog library.
add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
add_definitions("-DGL_GLEXT_PROTOTYPES")

add_executable(WICP main.cpp icp_3d.cc)
add_executable(pcl_icp pcl_icp.cpp)
# Link
target_link_libraries(WICP ${PCL_LIBRARIES} spdlog::spdlog)
target_link_libraries(pcl_icp ${PCL_LIBRARIES})

# 针对 MSVC 编译器添加 /bigobj 选项
if (MSVC)
    target_compile_options(WICP PRIVATE "/bigobj")
endif()
