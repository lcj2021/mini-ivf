cmake_minimum_required(VERSION 3.10)

project(Toy)

set(CMAKE_CXX_STANDARD 17)

add_library(toy SHARED  
    binary_io.hpp
    distance.hpp
    # hdf5_io.h
    index_ivf.cpp
    index_ivfpq.cpp
    kmeans.hpp
    quantizer.cpp
    util.cpp
)

target_compile_options(toy PUBLIC
    -Ofast
    -march=native
    -mtune=native
)

target_include_directories(toy PUBLIC
    $<BUILD_INTERFACE: ${PROJECT_SOURCE_DIR}>
)

target_link_libraries(toy PUBLIC
            # ${HDF5_CXX_LIBRARIES}
            # ${MPI_CXX_LIBRARIES}
            OpenMP::OpenMP_CXX
)