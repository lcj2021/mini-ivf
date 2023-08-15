#include "quantizer.h"
#include "kmeans.h"
#include <algorithm>
#include <tuple>
namespace Quantizer {


/**
 * @param 
*/

Quantizer::Quantizer(size_t D, size_t N, size_t M, size_t K, bool verbose=false)
    : D_(D), N_(N), M_(M), K_(K), verbose_(verbose)
{
    assert(D_ % M_ == 0);
    Ds_ = D_ / M_;

    if (M_ > 1 && K_ > 256) {
        std::cerr << "Error. K_ is too large. "
                  << "Currently, we only support PQ code with K_ <= 256 "
                  << "so that each subspace is represented by uint8_t (8 bit)"
                  << std::endl;
        throw;
    }

    centers_.clear();
    centers_.resize(M_, 
                    std::vector<std::vector<float>>(K_, 
                    std::vector<float>(Ds_)));
    assignments_.clear();
    assignments_.resize(M_, std::vector<int>(N_));
    assignments_.shrink_to_fit(); // If the previous fit malloced a long assignment array, shrink it.
}
/**
 * @param vec:  shape = Ds_
 * @param m:    m-th subspace
*/
int Quantizer::predict_one(const std::vector<float>& vec, size_t m)
{
    assert(vec.size() == Ds_);
    std::pair<size_t, float> nearest_one = nearest_center(vec, centers_[m]);
    return (int) nearest_one.first;
}

void
Quantizer::fit(const std::vector<float>& traindata, int iter = 20, int seed = 123) 
{
    assert(N_ == traindata.size() / D_);
    assert(K_ < N_ && "the number of training vector should be more than K_");

    // srand(seed);
    if (verbose_) {
        printf("N_: %zu, M_: %zu, K_: %zu\n", N_, M_, K_);
        std::cout << "iter: " << iter << ", seed: " << seed << std::endl;
    }

    auto traindata_trim = traindata;
    // traindata_trim.resize(std::min((size_t)100'000, N_) * D_);
    size_t Nt = traindata_trim.size() / D_;
    std::vector<std::vector<float>> train_vecs(Nt, std::vector<float>(D_));
    for (std::size_t n = 0; n < Nt; ++n) {
        std::copy(traindata_trim.begin() + n * D_, traindata_trim.begin() + (n + 1) * D_, 
                    train_vecs[n].begin());
    }

    // Perform k-means iterations for each subspace
    for (int m = 0; m < M_; ++m) {
        if (verbose_) {
            std::cout << "Training the subspace: " << m << " / " << M_ << std::endl;
        }
        std::vector<std::vector<float>> vecs_sub(Nt, std::vector<float>(Ds_, 0.0));
        for (int i = 0; i < Nt; ++i) {
            std::copy(train_vecs[i].begin() + m * Ds_, train_vecs[i].begin() + (m + 1) * Ds_, vecs_sub[i].begin());
        }
        std::vector<std::vector<float>> centroids;
        std::vector<int> labels;
        std::tie(centroids, labels) = KMeans(vecs_sub, K_, iter, "points");
        
        for (int k = 0; k < K_; ++k) {
            std::copy(centroids[k].begin(), centroids[k].end(), centers_[m][k].begin());
        }
        std::copy(labels.begin(), labels.end(), assignments_[m].begin());
    }
}

const std::vector<std::vector<int>>&
Quantizer::get_assignments() {return assignments_;}

const std::vector<std::vector<std::vector<float>>>&
Quantizer::get_centroids() {return centers_;}

void 
Quantizer::set_centroids(const std::vector<std::vector<std::vector<float>>>& centers_new)
{
    assert(centers_new.size() == M_);
    centers_ = centers_new;
}

std::vector<std::vector<uint8_t>> 
Quantizer::encode(const std::vector<std::vector<float>>& rawdata) 
{
    size_t N = rawdata.size();
    assert(D_ == rawdata[0].size());

    std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M_, 0));

    for (size_t m = 0; m < M_; ++m) {
        if (verbose_) {
            std::cout << "Encoding the subspace: " << m << " / " << M_ << std::endl;
        }
        std::vector<std::vector<float>> vecs_sub(N, std::vector<float>(Ds_));
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            std::copy_n(rawdata[i].begin() + m * Ds_, Ds_, vecs_sub[i].begin());
        }
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            auto [min_idx, min_dist] = nearest_center(vecs_sub[i], centers_[m]);
            codes[i][m] = (uint8_t)min_idx;
        }
    }
    return codes;
}

std::vector<std::vector<uint8_t>> 
Quantizer::encode(const std::vector<float>& rawdata) 
{
    size_t N = rawdata.size() / D_;

    std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M_, 0));

    for (size_t m = 0; m < M_; ++m) {
        if (verbose_) {
            std::cout << "Encoding the subspace: " << m << " / " << M_ << std::endl;
        }
        std::vector<std::vector<float>> vecs_sub(N, std::vector<float>(Ds_));
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            std::copy_n(rawdata.begin() + i * D_ + m * Ds_, Ds_, vecs_sub[i].begin());
        }
#pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            auto [min_idx, min_dist] = nearest_center(vecs_sub[i], centers_[m]);
            codes[i][m] = (uint8_t)min_idx;
        }
    }
    return codes;
}

std::vector<float> Quantizer::nth_vector(const std::vector<float>& long_code, size_t n)
{
    return std::vector<float>(long_code.begin() + n * D_, long_code.begin() + (n + 1) * D_);
}

// Each code: D = M_ * Ds_
std::vector<float>
Quantizer::nth_vector_mth_element(const std::vector<float>& long_code, size_t n, int m)
{
    return std::vector<float>(long_code.begin() + n * D_ + m * Ds_, 
                            long_code.begin() + n * D_ + (m + 1) * Ds_);
}



} // namespace pqkmeans
