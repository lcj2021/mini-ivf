#ifndef INCLUDE_QUANTIZER_HPP
#define INCLUDE_QUANTIZER_HPP

#include <vector>
#include <string>
#include <index.hpp>

namespace index {


template < typename vector_dimension_t >
class Quantizer {
public:
    explicit Quantizer(size_t D, size_t M, size_t K);

    /// @brief Pick a best one for a vector
    inline cluster_id_t PredictOne(const std::vector<vector_dimension_t> & vec, size_t m);
    
    void Fit(std::vector<vector_dimension_t> rawdata, size_t iter, int seed);    // pydata.shape == N * D
    
    void Load(const std::string & quantizer_filename);

    void Write(const std::string & quantizer_filename) const;

    const std::vector<std::vector<std::vector<vector_dimension_t>>> & GetCentroids() const;

    // const std::vector<std::vector<uint8_t>> & GetAssignments() const;

    std::vector<std::vector<uint8_t>> Encode(
        const std::vector<vector_dimension_t> & rawdata
    );

    std::vector<std::vector<uint8_t>> Encode(
        const std::vector<std::vector<vector_dimension_t>> & rawdata
    );

    bool Ready();

private:
    size_t D_;  // the demension of each vector
    size_t M_;  // the number of subspace
    size_t K_;  // the number of centroid for each subspace
    size_t Ds_; // the length/demension of (vector) each subspace

    // centers for clustering. shape = M_ * K_ * Ds_
    std::vector<std::vector<std::vector<vector_dimension_t>>> centers_;  
    // assignement for each intput vector. shape = M_ * N
    // assignments_[m][n] in [0, K_)
    // std::vector<std::vector<uint8_t>> assignments_;  

    bool ready_;

};


template class Quantizer<uint8_t>;
template class Quantizer<float>;



};

#endif