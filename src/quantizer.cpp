#include <quantizer.hpp>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <partition.hpp>
#include <vector_io.hpp>
#include <resize.hpp>

namespace index {


template <typename vector_dimension_t> 
Quantizer<vector_dimension_t>::Quantizer(size_t D, size_t M, size_t K):
    D_(D), M_(M), K_(K)
{
    assert( D_ % M_ == 0 );
    Ds_ = D_ / M_;

    if (M_ > 1 && K_ > 256) {
        std::cerr << "Error. K_ is too large. "
                  << "Currently, we only support PQ code with K_ <= 256 "
                  << "so that each subspace can be encoded into uint8_t (8 bit)"
                  << std::endl;
        throw;
    }

    centers_.clear();
    centers_.resize(M_, std::vector<std::vector<vector_dimension_t>>(K_, std::vector<vector_dimension_t>(Ds_)));

    assignments_.clear();
    assignments_.resize(M_, std::vector<uint8_t>(N_));
    assignments_.shrink_to_fit();

}



/**
 * @param vec:  shape = Ds_
 * @param m:    m-th subspace
*/
template <typename vector_dimension_t> inline cluster_id_t 
Quantizer<vector_dimension_t>::PrediceOne(const std::vector<vector_dimension_t> & vec, size_t m)
{
    return Partition<vector_dimension_t>::NearestCenter(vec, centers_[m]).first;
}



template <typename vector_dimension_t> void
Quantizer<vector_dimension_t>::Fit(std::vector<vector_dimension_t> rawdata, size_t iter, int seed)
{
    N = rawdata.size() / D_;
    assert(K_ < N_ && "the number of training vector should be more than K_");

    for (size_t m = 0; m < M_; m++)
    {
        std::vector<std::vector<vector_dimension_t>> vecs_sub(N, std::vector<vector_dimension_t>(Ds_));
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++)
        {
            std::copy_n(rawdata.begin() + i * D_ + m * Ds_, Ds_, vecs_sub[i].begin());
        }

        auto [centroids, labels] = Partition<vector_dimension_t>::KMeans(vecs_sub, K_, iter, "points");

        #pragma omp parallel for
        for (size_t k = 0; k < K_; k++)
        {
            std::copy(centroids[k].begin(), centroids[k].end(), centers_[m][k].begin());
        }
        std::copy(labels.begin(), labels.end(), assignments_[m].begin());
    }
}



template <typename vector_dimension_t> const std::vector<std::vector<std::vector<vector_dimension_t>>> &
Quantizer<vector_dimension_t>::GetCentroids() const { return centers_; }



template <typename vector_dimension_t> const std::vector<std::vector<uint8_t>> &
Quantizer<vector_dimension_t>::GetAssignments() const { return assignments_; }



template <typename vector_dimension_t> void
Quantizer<vector_dimension_t>::Load(const std::string & quantizer_path)
{
    const std::string center_suffix = "centers.fvecs";

    std::vector<vector_dimension_t> flat_center;
    utils::VectorIO<vector_dimension_t>::LoadFromFile(flat_center, quantizer_path + center_suffix);
    centers_ = utls::Resize<vector_dimension_t>::Nest(flat_center, M_, K_, Ds_);
}



template <typename vector_dimension_t> void
Quantizer<vector_dimension_t>::Write(const std::string & quantizer_path) const
{
    const std::string center_suffix = "centers.fvecs";
    auto flat_center = utils::Resize<vector_dimension_t>::Flatten(centers_);
    utils::VectorIO<vector_dimension_t>::WriteToFile(flat_center, {1, M_ * K_ * Ds_}, quantizer_path + center_suffix);
}



template <typename vector_dimension_t> std::vector<std::vector<uint8_t>>
Quantizer<vector_dimension_t>::Encode(const std::vector<vector_dimension_t> & rawdata)
{
    size_t N = rawdata.size();
    std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M_, 0x00));
    
    for (size_t m = 0; m < M_; m++)
    {
        std::vector<std::vector<uint8_t>> vecs_sub(N, std::vector<uint8_t>(Ds_));

        #pragma omp parallel for
        for (size_t i = 0; i < N; i++)
        {
            std::copy_n(rawdata.begin() + i * D_ + m * Ds_, Ds_, vecs_sub[i].begin());
        }

        #pragma omp parallel for
        for (size_t i = 0; i < N; i++)
        {
            auto [min_idx, min_dist] = utils::Partition<vector_dimension_t>::NearestCenter(vecs_sub[i], centers_[m]);
            codes[i][m] = (uint8_t)min_idx;
        }
    }

    return codes;
}



template <typename vector_dimension_t> std::vector<std::vector<uint8_t>>
Quantizer<vector_dimension_t>::Encode(const std::vector<std::vector<vector_dimension_t>> & rawdata)
{
    size_t N = rawdata.size();
    assert( D_ == rawdata[0].size() );

    std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M_, 0x00));

    for (size_t m = 0; m < M_; m++)
    {
        std::vector<std::vector<vector_dimension_t>> vecs_sub(N, std::vector<vector_dimension_t>(Ds_));

        #pragma omp parallel for
        for (size_t i = 0; i < N; i++)
        {
            std::copy_n(rawdata[i].begin() + m * Ds_, Ds_, vecs_sub[i].begin());
        }

        #pragma omp parallel for
        for (size_t i = 0; i < N; i++)
        {
            auto [min_idx, min_dist] = utils::Partition<vector_dimension_t>::NearestCenter(vecs_sub[i], centers_[m]);
            codes[i][m] = (uint8_t)min_idx;
        }
    }

    return codes;
}




};