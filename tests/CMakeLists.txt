cmake_minimum_required(VERSION 3.10)

project(Toy)

set(CMAKE_CXX_STANDARD 17)

set(TEST_SOURCES
    test_binary_io.cpp
    test_hdf5_io.cpp
    # test_deep10M.cpp
    test_ivf.cpp
    test_ivfpq.cpp
    test_ivfpq_gist1m_baseline.cpp
    test_ivfpq_sift1m_baseline.cpp
    test_ivfpq_sift1m_pred.cpp
    test_ivfpq_gist1m_obs.cpp
    test_ivfpq_sift1m_obs.cpp
    test_ivf_sift1m_baseline.cpp
    test_ivf_sift1m_pred.cpp
    test_rii.cpp
    test_rii_pred_sift1m.cpp
    test_rii_sift1m.cpp
    get_ivfpq_trainset.cpp
    get_groundtruth.cpp)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    message("${TEST_NAME} <- ${TEST_SOURCE}")
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PUBLIC toy)
    target_compile_options(${TEST_NAME} PUBLIC 
            -O3
            -march=native
            -mtune=native)
endforeach()



