#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.h"
#include <assert.h>
#include <algorithm>
#include <iomanip>

int main() {
    // std::vector<float> test_read_fvec;
    // load_from_file_binary(test_read_fvec, "../../dataset/sift/sift_learn.fvecs");

    // std::vector<int> test_read_ivec;
    // // load_from_file_binary(test_read_ivec, "../../dataset/sift/sift_query_groundtruth.ivecs");

    // std::vector<int> test_write_ivec = {1, 1, 4, 5, 1, 4};
    // write_to_file_binary(test_write_ivec, {2, 3}, "../../dataset/test_write_ivec.ivecs");

    // load_from_file_binary(test_read_ivec, "../../dataset/test_write_ivec.ivecs");
    // assert(test_write_ivec == test_read_ivec);

    // std::vector<unsigned char> test_read_sift1b;
    // load_from_file_binary(test_read_sift1b, "../../dataset/siftbig/bigann_base.bvecs", 500'000'000);

    // std::vector<float> test_read_deep;
    // load_from_file_binary_deep(test_read_deep, "../../dataset/deep/deep1b/base.1B.fbin.crop_nb_1000000", 1'000'000);
    // std::cout << *std::min_element(test_read_deep.begin(), test_read_deep.end()) << ' ' << *std::max_element(test_read_deep.begin(), test_read_deep.end()) << '\n';

    // std::vector<int> test_read_sift_gt;
    // load_from_file_binary(test_read_sift_gt, "../../dataset/sift/sift_train_groundtruth.ivecs");

    // std::vector<float> test_read_sift_d;
    // load_from_file_binary(test_read_sift_d, "../../dataset/sift/sift_train_distance.ivecs");

    std::vector<float> test_read_l, test_read_r, test_read_distance;
    load_from_file_binary(test_read_l, "./sift1m_l.fvecs");
    load_from_file_binary(test_read_r, "./sift1m_r.fvecs");
    load_from_file_binary(test_read_distance, "./sift1m_distance.fvecs");
    std::cout << std::fixed << std::setprecision(2);
    puts("l: ");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 6; ++j)
            std::cout << test_read_l[i * 6 + j] << " \n"[j == 5];
    puts("r: ");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 6; ++j)
            std::cout << test_read_r[i * 6 + j] << " \n"[j == 5];
    puts("distance: ");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 6; ++j)
            std::cout << test_read_distance[i * 6 + j] << " \n"[j == 5];

    std::vector<float> test_read_farthest;
    std::vector<int> test_read_distribution;
    load_from_file_binary(test_read_farthest, "./sift1m_farthest.fvecs");
    load_from_file_binary(test_read_distribution, "./sift1m_distribution.fvecs");
    puts("farthest: ");
    for (size_t i = 0; i < 5; ++i)
        std::cout << test_read_farthest[i] << " \n"[i == 4];
    puts("distribution: ");
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 20; ++j)
            std::cout << test_read_distribution[i * 20 + j] << " \n"[j == 19];

    std::vector<int> test_read_querycodes, test_read_centroidcodes;
    load_from_file_binary(test_read_querycodes, "./sift1m_querycodes.ivecs");
    load_from_file_binary(test_read_centroidcodes, "./sift1m_centroidcodes.ivecs");
    puts("querycodes: ");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 64; ++j)
            std::cout << (int)test_read_querycodes[i * 64 + j] << " \n"[j == 63];
    puts("centroidcodes: ");
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 64; ++j)
            std::cout << (int)test_read_centroidcodes[i * 64 + j] << " \n"[j == 63];
    return 0;
}