CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(Toy)

SET(CMAKE_CXX_STANDARD 17)

SET(TOY_ROOT ${CMAKE_SOURCE_DIR})

# SET(HDF5_ROOT "/var/lib/docker/lcj_ym_IR/hdf5/install_path")
# handle fatal error: H5Cpp.h: No such file or directory
# include <H5Cpp.h>
FIND_PACKAGE(HDF5 REQUIRED COMPONENTS CXX)
if (NOT HDF5_FOUND)
	MESSAGE(WARNING "hdf5 library not found, some tests will not be run")
else()
    INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
endif()

# SET(OpenMP "/var/lib/docker/lcj_graph_partitioning/mpi/openmpi-1.4.5")
FIND_PACKAGE(OpenMP REQUIRED)
INCLUDE_DIRECTORIES(${OpenMP_CXX_INCLUDE_DIRS})



FILE(GLOB TOY_SRC "${TOY_ROOT}/src/*.cpp")
INCLUDE_DIRECTORIES(${TOY_ROOT}/include)

ADD_LIBRARY(
    toy 
    SHARED  
    ${TOY_SRC}
)

target_include_directories(toy PUBLIC
    $<BUILD_INTERFACE:${TOY_INCLUDES}>)

TARGET_COMPILE_OPTIONS(
    toy
    PUBLIC
    -Ofast
    -march=native
    -mtune=native
)

TARGET_LINK_LIBRARIES(
    toy 
    PUBLIC
    # ${HDF5_CXX_LIBRARIES}
    # ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
)


# ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(toolkits)