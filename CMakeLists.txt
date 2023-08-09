
cmake_minimum_required(VERSION 3.10)

project(Toy)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

add_executable(main 
    src/main.cpp
    src/index_ivfpq.h
    src/pqkmeans.cpp
    src/util.h
    src/distance.h
    src/product_quantizer.h
    )

target_compile_options(main PRIVATE
    -Ofast
    -march=native
    -mtune=native
)

target_link_libraries(main PRIVATE
    OpenMP::OpenMP_CXX
)
