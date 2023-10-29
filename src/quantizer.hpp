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

namespace Quantizer {

class Quantizer {
public:
    Quantizer(size_t D, size_t N, size_t M, size_t K, bool verbose);

    int predict_one(const std::vector<float>& vec, size_t m);
    void fit(const std::vector<float>& rawdata, int iter, int seed);    // pydata.shape == N * D
    const std::vector<std::vector<int>>& get_assignments();

    void set_centroids(const std::vector<std::vector<std::vector<float>>>& centers_new);
    void Load(std::string quantizer_path);
    void Write(std::string quantizer_path);
    const std::vector<std::vector<std::vector<float>>>& get_centroids();
    std::vector<std::vector<uint8_t>> Encode(const std::vector<float>& rawdata);
    std::vector<std::vector<uint8_t>> Encode(const std::vector<std::vector<float>>& rawdata);

private:
    size_t D_;  // the demension of each vector
    size_t M_;  // the number of subspace
    size_t K_;  // the number of centroid for each subspace
    size_t Ds_; // the length/demension of (vector) each subspace
    size_t N_;  // the number of input rawdata (vector)
    bool verbose_;

    // centers for clustering. shape = M_ * K_ * Ds_
    std::vector<std::vector<std::vector<float>>> centers_;  
    // assignement for each intput vector. shape = M_ * N
    // assignments_[m][n] in [0, K_)
    std::vector<std::vector<int>> assignments_;  

    // Given a long (N * M) codes, pick up n-th code
    std::vector<float> NthVector(const std::vector<float>& long_code, size_t n);

    // Given a long (N * M) codes, pick up m-th element from n-th code
    std::vector<float> NthVectorMthElement(const std::vector<float>& long_code, size_t n, int m);

};

} // namespace Quantizer


#endif // QUANTIZER_H
