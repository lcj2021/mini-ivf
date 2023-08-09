#include <cstdint>
#include <vector>
#include <iostream>
#include "assert.h"

class ProductQuantizer {
public:
    ProductQuantizer(int M, int nbits, bool verbose=true) : M(M), nbits(nbits) {
        this->Ks = 1 << nbits;
        this->verbose = verbose;
        // this->code_dtype = (Ks <= 256) ? uint_8 : ((Ks <= 65536) ? uint_16 : uint_32);
        if (verbose) {
            std::cout << "M: " << M << ", Ks: " << Ks << '\n';
        }
    }

    auto& fit(const std::vector<std::vector<float>> &vecs, int iter = 20, int seed = 123) {
        assert(vecs[0].size() % this->M == 0);
        int N = vecs.size();
        int D = vecs[0].size();
        assert(this->Ks < N && "the number of training vector should be more than Ks");
        this->Ds = D / this->M;

        srand(seed);
        if (this->verbose) {
            std::cout << "iter: " << iter << ", seed: " << seed << std::endl;
        }

        // Allocate memory for codewords
        this->codewords = std::vector<std::vector<std::vector<float>>>(this->M,
                                                                       std::vector<std::vector<float>>(this->Ks,
                                                                                                      std::vector<float>(this->Ds)));

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
            std::tie(centroids, labels) = kmeans2(vecs_sub, Ks, iter, "points");

            for (int k = 0; k < Ks; ++k) {
                std::copy(centroids[k].begin(), centroids[k].end(), codewords[m][k].begin());
            }
        }

        return this->codewords;
    }

    std::vector<std::vector<unsigned char>> encode(const std::vector<std::vector<float>>& vecs) {
        std::size_t N = vecs.size();
        std::size_t D = vecs[0].size();
        assert(D == Ds * M);

        std::vector<std::vector<unsigned char>> codes(N, std::vector<unsigned char>(M, 0));

        for (std::size_t m = 0; m < M; ++m) {
            if (verbose) {
                std::cout << "Encoding the subspace: " << m + 1 << " / " << M << std::endl;
            }
            std::vector<std::vector<float>> vecs_sub(N, std::vector<float>(Ds));
            for (std::size_t i = 0; i < N; ++i) {
                std::copy_n(vecs[i].begin() + m * Ds, Ds, vecs_sub[i].begin());
            }
            for (std::size_t i = 0; i < N; ++i) {
                std::vector<float>& vec = vecs_sub[i];
                unsigned char min_idx = 0;
                float min_dist = std::numeric_limits<float>::max();
                for (std::size_t ks = 0; ks < Ks; ++ks) {
                    float dist = compute_distance(vec, codewords[m][ks]);
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

    float compute_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        float distance = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            distance += std::pow(vec1[i] - vec2[i], 2);
        }
        return std::sqrt(distance);
    }

    // kmeans2 implementation
    std::tuple<std::vector<std::vector<float>>, std::vector<int>> kmeans2(const std::vector<std::vector<float>>& obs, int k, int iter, const std::string& minit) {
        int n = obs.size();
        int dim = obs[0].size();

        // Initialize centroids based on minit
        std::vector<std::vector<float>> centroids(k, std::vector<float>(dim, 0.0));
        if (minit == "points") {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, n - 1);

            std::vector<int> initial_indices(k);
            for (int i = 0; i < k; ++i) {
                int idx = dist(gen);
                initial_indices[i] = idx;
                centroids[i] = obs[idx];
            }
        }

        // Perform k-means iterations
        std::vector<int> labels(n);
        for (int iter_count = 0; iter_count < iter; ++iter_count) {
            // std::cout << iter_count << '\n';
            // Assign each observation to the nearest centroid
            for (int i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int nearest_centroid = -1;
                for (int j = 0; j < k; ++j) {
                    float dist = compute_distance(obs[i], centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_centroid = j;
                    }
                }
                labels[i] = nearest_centroid;
            }

            // Update centroids based on assigned observations
            for (int j = 0; j < k; ++j) {
                std::vector<float> sum(dim, 0.0);
                int count = 0;
                for (int i = 0; i < n; ++i) {
                    if (labels[i] == j) {
                        std::transform(sum.begin(), sum.end(), obs[i].begin(), sum.begin(), std::plus<float>());
                        ++count;
                    }
                }
                if (count > 0) {
                    std::transform(sum.begin(), sum.end(), centroids[j].begin(), [count](float val) { return val / count; });
                }
            }
        }

        return std::make_tuple(centroids, labels);
    }

private:
    size_t M;       // number of subquantizers (sub-spaces)
    size_t nbits;   // nbits per quantization index 

    // derived from above
    size_t Ks;      // number of centroids for each subquantizer
    size_t Ds;      // number of demensions per subquantizers (sub-spaces)
    bool verbose;
    // std::size_t code_dtype;
    std::vector<std::vector<std::vector<float>>> codewords;

    int nearest_codeword_index(const std::vector<float> &vec, const std::vector<std::vector<float>> &codewords) {
        // Calculate distance and find the nearest codeword
        int nearest = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (int ks = 0; ks < this->Ks; ++ks) {
            float dist = 0.0;
            for (int j = 0; j < this->Ds; ++j) {
                dist += std::pow(vec[j] - codewords[ks][j], 2);
            }
            if (dist < min_dist) {
                nearest = ks;
                min_dist = dist;
            }
        }
        return nearest;
    }
};