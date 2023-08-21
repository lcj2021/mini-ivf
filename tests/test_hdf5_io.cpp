#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "hdf5_io.h"
#include <assert.h>

int main() {
    // std::vector<float> database;
    // auto [nb, D] = load_from_file_hdf5(database, "../../dataset/sift-128-euclidean.hdf5", "train");

    // std::vector<float> query;
    // auto [nq, _] = load_from_file_hdf5(query, "../../dataset/sift-128-euclidean.hdf5", "test");

    // std::vector<int> gt;
    // load_from_file_hdf5(gt, "../../dataset/sift-128-euclidean.hdf5", "neighbors");

    system("rm -rf ../../dataset/test_hdf5_io.hdf5");

    std::vector<int> test_neighbors = {1, 1, 4, 5, 1, 4};
    write_to_file_hdf5(test_neighbors, {2, 3}, "../../dataset/test_hdf5_io.hdf5", "neighbors");

    std::vector<int> test_read_neighbors;
    auto [r, c] = load_from_file_hdf5(test_read_neighbors, "../../dataset/test_hdf5_io.hdf5", "neighbors");

    for (const auto& x : test_read_neighbors) {
        std::cout << x << ' ';
    } std::cout << '\n';

    assert(test_neighbors == test_read_neighbors);

    return 0;
}