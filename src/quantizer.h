#ifndef QUANTIZER_H
#define QUANTIZER_H

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <cfloat>
#include "distance.h"

namespace Quantizer {

class Quantizer {
public:
    Quantizer(size_t D, size_t N, size_t M, size_t K, int itr, bool verbose);

    int predict_one(const std::vector<float> &vec, size_t m);
    void fit(const std::vector<float> &rawdata, int iter, int seed);    // pydata.shape == N * D
    const std::vector<std::vector<int>> GetAssignments();

    void SetClusterCenters(const std::vector<std::vector<std::vector<float>>> &centers_new);
    const std::vector<std::vector<std::vector<float>>>& GetClusterCenters();
    std::vector<std::vector<uint8_t>> encode(const std::vector<float>& rawdata);
    std::vector<std::vector<uint8_t>> encode(const std::vector<std::vector<float>>& rawdata);

private:
    size_t D_;  // the demension of each vector
    size_t M_;  // the number of subspace
    size_t K_;  // the number of centroid for each subspace
    size_t Ds_; // the length/demension of (vector) each subspace
    size_t N_;  // the number of input rawdata (vector)
    int iteration_;
    bool verbose_;

    // centers for clustering. shape = M_ * K_ * Ds_
    std::vector<std::vector<std::vector<float>>> centers_;  
    // assignement for each intput vector. shape = M_ * N
    // assignments_[m][n] in [0, K_]
    std::vector<std::vector<int>> assignments_;  

    // Given a long (N * M) codes, pick up n-th code
    std::vector<float> NthCode(const std::vector<float> &long_code, size_t n);

    // Given a long (N * M) codes, pick up m-th element from n-th code
    std::vector<float> NthCodeMthElement(const std::vector<float> &long_code, size_t n, int m);

};

} // namespace Quantizer


#endif // QUANTIZER_H