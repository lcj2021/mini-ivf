
cmake_minimum_required(VERSION 3.10)

project(Toy)

set(CMAKE_CXX_STANDARD 17)

add_executable(test_rii 
    src/test_rii.cpp
    src/index_rii.h
    src/pqkmeans.cpp
    src/quantizer.cpp
)

add_executable(test_ivfpq
    src/test_ivfpq.cpp
    src/index_ivfpq.h
    src/pqkmeans.cpp
    src/quantizer.cpp
)

add_executable(test_sift1m 
    src/test_sift1m.cpp
    src/index_rii.h
    src/pqkmeans.cpp
    src/quantizer.cpp
)

target_compile_options(test_ivfpq PRIVATE
    -Ofast
    -march=native
    -mtune=native
)

target_compile_options(test_rii PRIVATE
    -Ofast
    -march=native
    -mtune=native
)

target_compile_options(test_sift1m PRIVATE
    -Ofast
    -march=native
    -mtune=native
)


set(MPI "/var/lib/docker/lcj_graph_partitioning/mpi/openmpi-1.4.5")
find_package(MPI REQUIRED)

if(MPI_CXX_FOUND)
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
else()
    message(WARNING "MPI NOT found")
endif()

# set(HDF5_ROOT "/var/lib/docker/lcj_ym_IR/hdf5/install_path")
find_package(HDF5 REQUIRED COMPONENTS CXX)
if (NOT HDF5_FOUND)
	message(WARNING "hdf5 library not found, some tests will not be run")
else()
    include_directories(${HDF5_INCLUDE_DIR})
endif()


target_link_libraries(test_rii PRIVATE
    ${MPI_CXX_LIBRARIES}
)

target_link_libraries(test_ivfpq PRIVATE
    ${MPI_CXX_LIBRARIES}
)

target_link_libraries(test_sift1m PRIVATE
    ${HDF5_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
)

message("HDF5 LIB: ${HDF5_LIBRARIES}")
message("HDF5 INC: ${HDF5_INCLUDE_DIRS}")

