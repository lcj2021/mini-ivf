#ifndef INCLUDE_QUANTIZER_HPP
#define INCLUDE_QUANTIZER_HPP

#include <vector>
#include <string>
#include <index.hpp>

namespace index {


template < typename vector_dimension_t >
class Quantizer {
public:
    explicit Quantizer(size_t D, size_t N, size_t M, size_t K);

    /// @brief Pick a best one for a vector
    inline cluster_id_t PrediceOne(const std::vector<vector_dimension_t> & vec, size_t m);
    
    void Fit(std::vector<vector_dimension_t> rawdata, size_t iter, int seed);    // pydata.shape == N * D
    
    void Load(const std::string & quantizer_path);

    void Write(const std::string & quantizer_path) const;

    const std::vector<std::vector<std::vector<vector_dimension_t>>> & GetCentroids() const;

    const std::vector<std::vector<uint8_t>> & GetAssignments() const;

    std::vector<std::vector<uint8_t>> Encode(
        const std::vector<vector_dimension_t> & rawdata
    );

    std::vector<std::vector<uint8_t>> Encode(
        const std::vector<std::vector<vector_dimension_t>> & rawdata
    );

private:
    size_t D_;  // the demension of each vector
    size_t M_;  // the number of subspace
    size_t K_;  // the number of centroid for each subspace
    size_t Ds_; // the length/demension of (vector) each subspace
    size_t N_;  // the number of input rawdata (vector)

    // centers for clustering. shape = M_ * K_ * Ds_
    std::vector<std::vector<std::vector<vector_dimension_t>>> centers_;  
    // assignement for each intput vector. shape = M_ * N
    // assignments_[m][n] in [0, K_)
    std::vector<std::vector<uint8_t>> assignments_;  

    // Given a long (N * M) codes, pick up n-th code
    std::vector<float> NthVector(const std::vector<vector_dimension_t>& long_code, size_t n);

    // Given a long (N * M) codes, pick up m-th element from n-th code
    std::vector<float> NthVectorMthElement(const std::vector<vector_dimension_t>& long_code, size_t n, int m);

};

};

#endif