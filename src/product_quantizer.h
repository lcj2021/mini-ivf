#pragma once

#include <cstdint>
#include <iostream>
#include <algorithm>

#include "assert.h"
#include "distance.h"
#include "kmeans.h"

class ProductQuantizer {
public:
    ProductQuantizer(size_t D, size_t M, size_t nbits, bool verbose=true) : D(D), M(M), nbits(nbits), verbose(verbose) 
    {
        Ks = (size_t)1 << nbits;
        Ds = D / M;
        // code_dtype = (Ks <= 256) ? uint_8 : ((Ks <= 65536) ? uint_16 : uint_32);

        // Allocate memory for codewords
        codewords = std::vector<std::vector<std::vector<float>>>(M, \
                                std::vector<std::vector<float>>(Ks, \
                                std::vector<float>(Ds, 0)));

        if (verbose) {
            std::cout << "M: " << M << ", Ks: " << Ks << '\n';
        }
    }

    auto& fit(const std::vector<std::vector<float>> &vecs, int iter = 20, int seed = 123) 
    {
        assert(vecs[0].size() % M == 0);
        int N = vecs.size();
        int D = vecs[0].size();
        assert(Ks < N && "the number of training vector should be more than Ks");

        // srand(seed);
        if (verbose) {
            std::cout << "iter: " << iter << ", seed: " << seed << std::endl;
        }

        // Perform k-means iterations for each subspace
        for (int m = 0; m < M; ++m) {
            if (verbose) {
                std::cout << "Training the subspace: " << m << " / " << M << std::endl;
            }
            std::vector<std::vector<float>> vecs_sub(N, std::vector<float>(Ds, 0.0));
            for (int i = 0; i < N; ++i) {
                std::copy(vecs[i].begin() + m * Ds, vecs[i].begin() + (m + 1) * Ds, vecs_sub[i].begin());
            }
            
            std::vector<std::vector<float>> centroids;
            std::vector<int> labels;
            std::tie(centroids, labels) = Clustering::KMeans(vecs_sub, Ks, iter, "points");
            
            for (int k = 0; k < Ks; ++k) {
                std::copy(centroids[k].begin(), centroids[k].end(), codewords[m][k].begin());
            }
        }

        return codewords;
    }

    std::vector<std::vector<uint8_t>> encode(const std::vector<std::vector<float>>& vecs) 
    {
        std::size_t N = vecs.size();
        std::size_t D = vecs[0].size();
        assert(D == Ds * M);

        std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M, 0));

        for (std::size_t m = 0; m < M; ++m) {
            if (verbose) {
                std::cout << "Encoding the subspace: " << m << " / " << M << std::endl;
            }
            std::vector<std::vector<float>> vecs_sub(N, std::vector<float>(Ds));
            for (std::size_t i = 0; i < N; ++i) {
                std::copy_n(vecs[i].begin() + m * Ds, Ds, vecs_sub[i].begin());
            }
            for (std::size_t i = 0; i < N; ++i) {
                const std::vector<float>& vec = vecs_sub[i];
                uint8_t min_idx = 0;
                float min_dist = std::numeric_limits<float>::max();
                for (std::size_t ks = 0; ks < Ks; ++ks) {
                    float dist = Toy::fvec_L2sqr(vec.data(), codewords[m][ks].data(), vec.size());
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = ks;
                    }
                }
                codes[i][m] = min_idx;
            }
        }

        return codes;
    }

    auto& fit(const std::vector<float> &vecs, int iter = 20, int seed = 123) 
    {
        assert(D % M == 0);
        size_t N = vecs.size() / D;
        assert(Ks < N && "the number of training vector should be more than Ks");

        // srand(seed);
        if (verbose) {
            std::cout << "iter: " << iter << ", seed: " << seed << std::endl;
        }

        std::vector<size_t> ids(N);
        size_t Nt = std::min(N, (size_t)100'000);
        std::vector<std::vector<float>> train_vecs(Nt, std::vector<float>(D));
        std::iota(ids.begin(), ids.end(), 0); // 0, 1, 2, ..., codes.size()-1
        std::mt19937 default_random_engine(0);
        std::shuffle(ids.begin(), ids.end(), default_random_engine);
        for (std::size_t k = 0; k < Nt; ++k) {
            size_t id = ids[k];
            std::copy(vecs.begin() + id * D, vecs.begin() + (id + 1) * D, 
                        train_vecs[k].begin());
        }

        // Perform k-means iterations for each subspace
        for (int m = 0; m < M; ++m) {
            if (verbose) {
                std::cout << "Training the subspace: " << m << " / " << M << std::endl;
            }
            std::vector<std::vector<float>> vecs_sub(Nt, std::vector<float>(Ds, 0.0));
            for (int i = 0; i < Nt; ++i) {
                std::copy(train_vecs[i].begin() + m * Ds, train_vecs[i].begin() + (m + 1) * Ds, vecs_sub[i].begin());
            }
            
            std::vector<std::vector<float>> centroids;
            std::vector<int> labels;
            std::tie(centroids, labels) = Clustering::KMeans(vecs_sub, Ks, iter, "points");
            
            for (int k = 0; k < Ks; ++k) {
                std::copy(centroids[k].begin(), centroids[k].end(), codewords[m][k].begin());
            }
        }

        return codewords;
    }

    std::vector<std::vector<uint8_t>> encode(const std::vector<float>& vecs) 
    {
        std::size_t N = vecs.size() / D;
        assert(D == Ds * M);

        std::vector<std::vector<uint8_t>> codes(N, std::vector<uint8_t>(M, 0));
// #pragma omp parallel for
        for (std::size_t m = 0; m < M; ++m) {
            if (verbose) {
                std::cout << "Encoding the subspace: " << m << " / " << M << std::endl;
            }
            std::vector<std::vector<float>> vecs_sub(N, std::vector<float>(Ds));
            for (std::size_t i = 0; i < N; ++i) {
                std::copy_n(vecs.begin() + i * D + m * Ds, Ds, vecs_sub[i].begin());
            }
            for (std::size_t i = 0; i < N; ++i) {
                const std::vector<float>& vec = vecs_sub[i];
                uint8_t min_idx = 0;
                float min_dist = std::numeric_limits<float>::max();
                for (std::size_t ks = 0; ks < Ks; ++ks) {
                    float dist = Toy::fvec_L2sqr(vec.data(), codewords[m][ks].data(), vec.size());
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = ks;
                    }
                }
                codes[i][m] = min_idx;
            }
        }

        return codes;
    }

private:
    size_t D;
    size_t M;       // number of subquantizers (sub-spaces)
    size_t nbits;   // nbits per quantization index 

    // derived from above
    size_t Ks;      // number of centroids for each subquantizer
    size_t Ds;      // number of demensions per subquantizers (sub-spaces)
    bool verbose;
    // std::size_t code_dtype;
    std::vector<std::vector<std::vector<float>>> codewords;
};