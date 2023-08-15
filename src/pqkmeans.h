#ifndef PQKMEANS_PQKMEANS_H
#define PQKMEANS_PQKMEANS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <cfloat>
// #include "distance.h"

namespace pqkmeans {

class PQKMeans {
public:
    PQKMeans(std::vector<std::vector<std::vector<float>>> codewords, int K, int itr, bool verbose);

    int predict_one(const std::vector<uint8_t>& pyvector);
    void fit(const std::vector<uint8_t>& pydata);  // pydata is a long array. pydata.size == N * M

    const std::vector<int> get_assignments();

    void set_centroids(const std::vector<std::vector<uint8_t>>& centers_new);
    std::vector<std::vector<uint8_t>> get_centroids();
    int K() const { return K_; }
    int Iteration() const { return iteration_; }
    bool Verbose() const { return verbose_; }
    float SymmetricDistance(const std::vector<uint8_t>& code1,
                            const std::vector<uint8_t>& code2);

    float L2SquaredDistance(const std::vector<float>& vec1,
                            const std::vector<float>& vec2);
    // [m][k1][k2]: m-th subspace, the L2 squared distance between k1-th and k2-th codewords
    std::vector<std::vector<std::vector<float>>> distance_matrices_among_codewords_;

private:
    std::vector<std::vector<std::vector<float>>> codewords_;  // codewords for PQ encoding
    int K_;
    int iteration_;
    std::size_t M_; // the number of subspace
    bool verbose_;

    std::vector<std::vector<uint8_t>> centers_;  // centers for clustering.
    std::vector<int> assignments_;  // assignement for each intput vector



    void InitializeCentersByRandomPicking(const std::vector<uint8_t>& codes,  // codes.size == N * M
                                          int K,
                                          std::vector<std::vector<uint8_t>> *centers_);

    // Linear search by Symmetric Distance computation. Return the best one (id, distance)
    std::pair<std::size_t, float> FindNearetCenterLinear(const std::vector<uint8_t>& query,
                                                         const std::vector<std::vector<uint8_t>>& codes);

    // Compute a new cluster center from assigned codes. codes: All N codes. selected_ids: selected assigned ids.
    // e.g., If selected_ids=[4, 25, 13], then codes[4], codes[25], and codes[13] are averaged by the proposed sparse voting scheme.
    std::vector<uint8_t> ComputeCenterBySparseVoting(const std::vector<uint8_t>& codes,  // codes.size == N * M
                                                           const std::vector<std::size_t>& selected_ids);

    // Given a long (N * M) codes, pick up n-th code
    std::vector<uint8_t> nth_vector(const std::vector<uint8_t>& long_code, std::size_t n);

    // Given a long (N * M) codes, pick up m-th element from n-th code
    uint8_t nth_vector_mth_element(const std::vector<uint8_t>& long_code, std::size_t n, int m);

};

} // namespace pqkmeans


#endif // PQKMEANS_PQKMEANS_H
