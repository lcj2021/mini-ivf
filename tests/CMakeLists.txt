cmake_minimum_required(VERSION 3.10)

project(Toy)

set(CMAKE_CXX_STANDARD 17)

set(TEST_SOURCES
    test_binary_io.cpp
    # test_hdf5_io.cpp
    test_ivf.cpp
    test_ivfpq.cpp
    test_ivfpq_gist1m_baseline.cpp
    test_ivfpq_sift1m_baseline.cpp
    test_ivfpq_sift10m_baseline.cpp
    test_ivf_sift1m_baseline.cpp
    test_ivf_sift10m_baseline.cpp
    test_ivf_sift100m_baseline.cpp
    test_ivfpq_load_book.cpp
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    message("${TEST_NAME} <- ${TEST_SOURCE}")
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PUBLIC toy)
    target_compile_options(${TEST_NAME} PUBLIC 
            -Ofast
            -march=native
            -mtune=native
    )
endforeach()



