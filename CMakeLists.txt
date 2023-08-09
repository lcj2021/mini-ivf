
cmake_minimum_required(VERSION 3.10)

project(LRC)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

# add_subdirectory(src)
add_executable(main 
    src/main.cpp
    src/rii.h
    src/pqkmeans.cpp
    src/util.h
    src/distance.h
    src/product_quantizer.h
    # src/inverted_lists.cpp
    )

target_compile_options(main PRIVATE
    -Ofast
    -march=native
    -mtune=native
)

target_link_libraries(main PRIVATE
    OpenMP::OpenMP_CXX
)
