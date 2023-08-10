#pragma once

#include <vector>
#include <string>
#include <random>
#include "distance.h"

namespace Clustering {

    // kmeans Lloyd implementation
    std::tuple<std::vector<std::vector<float>>, std::vector<int>> KMeans(const std::vector<std::vector<float>>& obs, int k, int iter, const std::string& minit) {
        int n = obs.size();
        int dim = obs[0].size();

        // Initialize centroids based on minit
        std::vector<std::vector<float>> centroids(k, std::vector<float>(dim, 0.0));
        if (minit == "points") {
            std::default_random_engine rd;
            // std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, n - 1);

            std::vector<int> initial_indices(k);
            for (int i = 0; i < k; ++i) {
                int idx = dist(rd);
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
                    float dist = Toy::fvec_L2sqr(obs[i].data(), centroids[j].data(), obs[i].size());
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

        return {centroids, labels};
    }



    int nearest_codeword_index(const std::vector<float> &vec, const std::vector<std::vector<float>> &codewords) {
        // Calculate distance and find the nearest codeword
        int nearest = 0;
        size_t Ks = codewords.size(), Ds = codewords[0].size();
        float min_dist = std::numeric_limits<float>::max();
        for (int ks = 0; ks < Ks; ++ks) {
            float dist = Toy::fvec_L2sqr(vec.data(), codewords[ks].data(), Ds);
            if (dist < min_dist) {
                nearest = ks;
                min_dist = dist;
            }
        }
        return nearest;
    }
};