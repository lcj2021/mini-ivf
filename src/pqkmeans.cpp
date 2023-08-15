#include "pqkmeans.h"

namespace pqkmeans {


  

PQKMeans::PQKMeans(std::vector<std::vector<std::vector<float> > > codewords, int K, int itr, bool verbose)
    : codewords_(codewords), K_(K), iteration_(itr), verbose_(verbose)
{
    assert(!codewords.empty() && !codewords[0].empty() && !codewords[0][0].empty());
    M_ = codewords.size(); // The number of subspace
    size_t Ks = codewords[0].size();  // The number of codewords for each subspace

    if (256 < Ks) {
        std::cerr << "Error. Ks is too large. "
                  << "Currently, we only support PQ code with Ks <= 256 "
                  << "so that each subspace is represented by uint8_t (8 bit)"
                  << std::endl;
        throw;
    }

    // Compute distance-matrices among codewords
    distance_matrices_among_codewords_.resize(M_, 
                        std::vector<std::vector<float>>(Ks, 
                        std::vector<float>(Ks, 0)));

    size_t Ds = codewords[0][0].size();
    for (size_t m = 0; m < M_; ++m) {
        for (size_t k1 = 0; k1 < Ks; ++k1) {
            for (size_t k2 = 0; k2 < Ks; ++k2) {
                distance_matrices_among_codewords_[m][k1][k2] =
                        // Toy::fvec_L2sqr(codewords[m][k1].data(), codewords[m][k2].data(), Ds);
                        L2SquaredDistance(codewords[m][k1], codewords[m][k2]);
            }
        }
    }
}

int PQKMeans::predict_one(const std::vector<uint8_t>& pyvector)
{
    assert(pyvector.size() == M_);
    std::pair<size_t, float> nearest_one = FindNearetCenterLinear(pyvector, centers_);
    return (int) nearest_one.first;
}



void PQKMeans::fit(const std::vector<uint8_t>& pydata) {
    assert( (size_t) K_ * M_ <= pydata.size());
    assert(pydata.size() % M_ == 0);
    size_t N = pydata.size() / M_;

    // Refresh
    centers_.clear();
    centers_.resize((size_t) K_, std::vector<uint8_t>(M_));
    assignments_.clear();
    assignments_.resize(N);
    assignments_.shrink_to_fit(); // If the previous fit malloced a long assignment array, shrink it.

    std::vector<std::vector<uint8_t>> centers_new, centers_old;

    // (1) Initialization
    // [todo] Currently, only random pick is supported
    InitializeCentersByRandomPicking(pydata, K_, &centers_new);

    // selected_indices_foreach_center[k] has indices, where
    // each pydata[id] is assigned to k-th center.
    std::vector<std::vector<size_t>> selected_indices_foreach_center(K_);
    for (auto& selected_indices : selected_indices_foreach_center) {
        selected_indices.reserve( N / K_); // roughly allocate
    }

    std::vector<double> errors(N, 0);

    for (int itr = 0; itr < iteration_; ++itr) {
        if (verbose_) {
            std::cout << "Iteration start: " << itr << " / " << iteration_ << std::endl;
        }
        auto start = std::chrono::system_clock::now(); // ---- timer start ---

        centers_old = centers_new;

        // (2) Find NN centers
        selected_indices_foreach_center.clear();
        selected_indices_foreach_center.resize(K_);

        double error_sum = 0;

#pragma omp parallel for
        for(size_t n = 0; n < N; ++n) {
            std::pair<size_t, float> min_k_dist = FindNearetCenterLinear(nth_vector(pydata, n), centers_old);
            assignments_[n] = (int) min_k_dist.first;
            errors[n] = min_k_dist.second;
        }
        // (2.5) assignments -> selected_indices_foreach_center
        for (size_t n = 0; n < N; ++n) {
            int k = assignments_[n];
            selected_indices_foreach_center[k].emplace_back(n);
            error_sum += errors[n];
        }

        if (verbose_) {
            std::cout << "find_nn finished. Error: " << error_sum / N << std::endl;
            std::cout << "find_nn_time,"
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
                      << std::endl;
        }

        // (3) Compute centers
        if (itr != iteration_ - 1) {
            // Usually, centers would be updated.
            // After the last assignment, centers should not be updated, so this block is skiped.

            for (int k = 0; k < K_; ++k) {
                if (selected_indices_foreach_center[k].empty()) {
                    if (verbose_) {
                        std::cout << "Caution. No codes are assigned to " << k << "-th centers." << std::endl;
                    }
                    continue;
                }
                centers_new[k] =
                        ComputeCenterBySparseVoting(pydata, selected_indices_foreach_center[k]);
            }
        }
        if (verbose_) {
            std::cout << "find_nn+update_center_time,"
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
                      << std::endl;
        }

    }
    centers_ = centers_new;
}

const std::vector<int> PQKMeans::get_assignments()
{
    return assignments_;
}

std::vector<std::vector<uint8_t>> PQKMeans::get_centroids()
{
    return centers_;
}

void PQKMeans::set_centroids(const std::vector<std::vector<uint8_t>>& centers_new)
{
    assert(centers_new.size() == (size_t) K_);
    centers_ = centers_new;
}


float PQKMeans::SymmetricDistance(const std::vector<uint8_t>& code1,
                                  const std::vector<uint8_t>& code2)
{
    assert(code1.size() == code2.size());
    assert(code1.size() == M_);
    float dist = 0;
    for (size_t m = 0; m < M_; ++m) {
        dist += distance_matrices_among_codewords_[m][code1[m]][code2[m]];
    }
    return dist;
}

float PQKMeans::L2SquaredDistance(const std::vector<float>& vec1,
                                  const std::vector<float>& vec2)
{
    assert(vec1.size() == vec2.size());
    float dist = 0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return dist;
}



void PQKMeans::InitializeCentersByRandomPicking(const std::vector<uint8_t>& codes, int K, std::vector<std::vector<uint8_t> > *centers)
{
    assert(centers != nullptr);
    centers->clear();
    centers->resize(K);

    std::vector<int> ids(codes.size() / M_);
    std::iota(ids.begin(), ids.end(), 0); // 0, 1, 2, ..., codes.size()-1
    std::mt19937 default_random_engine(0);
    std::shuffle(ids.begin(), ids.end(), default_random_engine);
    for (size_t k = 0; k < (size_t) K; ++k) {
        (*centers)[k] = nth_vector(codes, ids[k]);
    }

}

std::pair<size_t, float> PQKMeans::FindNearetCenterLinear(const std::vector<uint8_t>& query,
                                                               const std::vector<std::vector<uint8_t> >& codes)
{
    std::vector<float> dists(codes.size());

    // Compute a distance from a query to each code in parallel
    size_t sz = codes.size();
#pragma omp parallel for
    for (size_t i = 0; i < sz; ++i) {
        dists[i] = SymmetricDistance(query, codes[i]);
    }

    // Just pick up the closest one
    float min_dist = FLT_MAX;
    int min_i = -1;
    for (size_t i = 0; i < sz; ++i) {
        if (dists[i] < min_dist) {
            min_i = static_cast<int>(i);
            min_dist = dists[i];
        }
    }
    assert(min_i != -1);

    return std::pair<size_t, float>((size_t)min_i, min_dist);
}




std::vector<uint8_t> PQKMeans::ComputeCenterBySparseVoting(const std::vector<uint8_t>& codes, 
                                                        const std::vector<size_t>& selected_ids)
{
    std::vector<uint8_t> average_code(M_);
    size_t Ks = codewords_[0].size();  // The number of codewords for each subspace

    for (size_t m = 0; m < M_; ++m) {
        // Scan the assigned codes, then create a freq-histogram
        std::vector<int> frequency_histogram(Ks, 0);
        for (const auto& id : selected_ids) {
            ++frequency_histogram[ nth_vector_mth_element(codes, id, m) ];
        }

        // Vote the freq-histo weighted by ditance matrices
        std::vector<float> vote(Ks, 0);
        for (size_t k1 = 0; k1 < Ks; ++k1) {
            int freq = frequency_histogram[k1];
            if (freq == 0) { // not assigned for k1. Skip it.
                continue;
            }
            for (size_t k2 = 0; k2 < Ks; ++k2) {
                vote[k2] += (float) freq * distance_matrices_among_codewords_[m][k1][k2];
            }
        }

        // find min
        float min_dist = FLT_MAX;
        int min_ks = -1;
        for (size_t ks = 0; ks < Ks; ++ks) {
            if (vote[ks] < min_dist) {
                min_ks = (int) ks;
                min_dist = vote[ks];
            }
        }
        assert(min_ks != -1);
        average_code[m] = (uint8_t) min_ks;
    }
    return average_code;
}

std::vector<uint8_t> PQKMeans::nth_vector(const std::vector<uint8_t>& long_code, size_t n)
{
    return std::vector<uint8_t>(long_code.begin() + n * M_, long_code.begin() + (n + 1) * M_);
}

uint8_t PQKMeans::nth_vector_mth_element(const std::vector<uint8_t>& long_code, size_t n, int m)
{
    return long_code[ n * M_ + m];
}



} // namespace pqkmeans
