cmake_minimum_required(VERSION 3.10)

project(Toy)

set(CMAKE_CXX_STANDARD 17)

# set(HDF5_ROOT "/var/lib/docker/lcj_ym_IR/hdf5/install_path")
# handle fatal error: H5Cpp.h: No such file or directory
# include <H5Cpp.h>
find_package(HDF5 REQUIRED COMPONENTS CXX)
if (NOT HDF5_FOUND)
	message(WARNING "hdf5 library not found, some tests will not be run")
else()
    include_directories(${HDF5_INCLUDE_DIR})
endif()

set(OpenMP "/var/lib/docker/lcj_graph_partitioning/mpi/openmpi-1.4.5")
find_package(OpenMP REQUIRED)
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(SRC_DIR "src")  
set(TESTS_DIR "tests")

add_subdirectory(src)
add_subdirectory(tests)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
