#include "quantizer.hpp"
#include "kmeans.hpp"
#include "binary_io.hpp"
#include "util.hpp"
#include <algorithm>
#include <tuple>
namespace Quantizer {


/**
 * @param 
*/

template <typename T>
Quantizer<T>::Quantizer(size_t D, size_t N, size_t M, size_t K, bool verbose)
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
template <typename T>
uint32_t Quantizer<T>::predict_one(const T* vec, uint32_t m)
{
    std::pair<uint32_t, float> nearest_one = NearestCenter(vec, centers_[m]);
    return nearest_one.first;
}

template <typename T>
void Quantizer<T>::fit(const std::vector<T>& traindata, int iter, int seed) 
{
    assert(N_ == traindata.size() / D_);
    assert(K_ < N_ && "the number of training vector should be more than K_");

    // srand(seed);
    if (verbose_) {
        printf("N_: %zu, M_: %zu, K_: %zu\n", N_, M_, K_);
        std::cout << "iter: " << iter << ", seed: " << seed << std::endl;
    }

    auto traindata_trim = traindata;
    size_t Nt = traindata_trim.size() / D_;
    std::vector<std::vector<T>> train_vecs(Nt, std::vector<T>(D_));

    #pragma omp parallel for
    for (size_t n = 0; n < Nt; ++n) {
        std::copy(traindata_trim.begin() + n * D_, traindata_trim.begin() + (n + 1) * D_, 
                    train_vecs[n].begin());
    }

    // Perform k-means iterations for each subspace
    for (int m = 0; m < M_; ++m) {
        if (verbose_) {
            std::cout << "Training the subspace: " << m << " / " << M_ << std::endl;
        }
        std::vector<std::vector<T>> vecs_sub(Nt, std::vector<T>(Ds_, 0.0));
        #pragma omp parallel for
        for (int i = 0; i < Nt; ++i) {
            std::copy(train_vecs[i].begin() + m * Ds_, train_vecs[i].begin() + (m + 1) * Ds_, 
                        vecs_sub[i].begin());
        }
        std::vector<std::vector<float>> centroids;
        std::vector<int> labels;
        std::tie(centroids, labels) = KMeans<T>(vecs_sub, K_, iter, "points");
        
        for (int k = 0; k < K_; ++k) {
            std::copy(centroids[k].begin(), centroids[k].end(), centers_[m][k].begin());
        }
        std::copy(labels.begin(), labels.end(), assignments_[m].begin());
    }
}

template <typename T>
const std::vector<std::vector<int>>&
Quantizer<T>::GetAssignments() {return assignments_;}

template <typename T>
const std::vector<std::vector<std::vector<float>>>&
Quantizer<T>::get_centroids() {return centers_;}

template <typename T>
void Quantizer<T>::SetCentroids(const std::vector<std::vector<std::vector<float>>>& centers_new)
{
    assert(centers_new.size() == M_);
    centers_ = centers_new;
}

template <typename T>
void Quantizer<T>::Load(std::string quantizer_path)
{
    std::string center_suffix = "centers.fvecs";
    std::string assign_suffix = "assignments.ivecs";
    std::vector<float> flat_center;
    // std::vector<int> flat_assign;
    LoadFromFileBinary<float>(flat_center, quantizer_path + center_suffix);
    // LoadFromFileBinary(flat_assign, quantizer_path + assign_suffix);
    this->centers_ = nest(flat_center, std::vector<size_t>{M_, K_, Ds_});
    // this->assignments_ = nest(flat_assign, std::vector<size_t>{M_, K_, Ds_});
    // LoadFromFileBinary(assignments_, quantizer_path + assign_suffix);
}

template <typename T>
void Quantizer<T>::Write(std::string quantizer_path)
{
    std::string center_suffix = "centers.fvecs";
    std::string assign_suffix = "assignments.ivecs";

    auto flat_center = flatten(this->centers_);
    // auto flat_assign = flatten(this->assignments_);
    WriteToFileBinary(flat_center, {1, M_ * K_ * Ds_}, quantizer_path + center_suffix);
    // WriteToFileBinary(assignments_, quantizer_path + assign_suffix);
}

template <typename T>
std::vector<std::vector<uint8_t>> 
Quantizer<T>::Encode(const std::vector<std::vector<T>>& rawdata) 
{
    size_t N = rawdata.size();
    assert(D_ == rawdata[0].size());

    std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M_, 0));

    for (size_t m = 0; m < M_; ++m) {
        if (verbose_) {
            std::cout << "Encoding the subspace: " << m << " / " << M_ << std::endl;
        }
        std::vector<std::vector<T>> vecs_sub(N, std::vector<T>(Ds_));

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            std::copy_n(rawdata[i].begin() + m * Ds_, Ds_, vecs_sub[i].begin());
        }

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            auto [min_idx, min_dist] = NearestCenter<T>(vecs_sub[i].data(), centers_[m]);
            codes[i][m] = (uint8_t)min_idx;
        }
    }
    return codes;
}

template <typename T>
std::vector<std::vector<uint8_t>> 
Quantizer<T>::Encode(const std::vector<T>& rawdata) 
{
    size_t N = rawdata.size() / D_;

    std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M_, 0));

    for (size_t m = 0; m < M_; ++m) {
        if (N > 1 && verbose_) {
            std::cout << "Encoding the subspace: " << m << " / " << M_ << std::endl;
        }
        std::vector<std::vector<T>> vecs_sub(N, std::vector<T>(Ds_));

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            std::copy_n(rawdata.begin() + i * D_ + m * Ds_, Ds_, vecs_sub[i].begin());
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            auto [min_idx, min_dist] = NearestCenter<T>(vecs_sub[i].data(), centers_[m]);
            codes[i][m] = (uint8_t)min_idx;
        }
    }
    return codes;
}

template <typename T>
std::vector<T> 
Quantizer<T>::NthVector(const std::vector<T>& long_code, size_t n)
{
    return std::vector<T>(long_code.begin() + n * D_, long_code.begin() + (n + 1) * D_);
}

// Each code: D = M_ * Ds_
template <typename T>
std::vector<T>
Quantizer<T>::NthVectorMthElement(const std::vector<T>& long_code, size_t n, int m)
{
    return std::vector<T>(long_code.begin() + n * D_ + m * Ds_, 
                            long_code.begin() + n * D_ + (m + 1) * Ds_);
}

template class Quantizer<uint8_t>;
template class Quantizer<float>;


} // namespace pqkmeans
