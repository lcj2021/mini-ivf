#include <partition.hpp>
#include <iostream>
#include <algorithm>

namespace index {

template<typename vector_dimension_t>
std::pair<cluster_id_t, float> Partition<vector_dimension_t>::NearestCenter(
    const std::vector<vector_dimension_t> & query, 
    const std::vector<std::vector<vector_dimension_t>> & centers
)
{
    size_t k = centers.size(), ds = centers[0].size();
    assert(k);
    std::vector<float> dists(k);

    /// @bug No omp here.
    // #pragema omp parallel for
    for (size_t i = 0; i < k; i++)
    {
        dists[i] = vec_L2sqr(query.data(), centers[i].data(), ds);
    }

    // Just pick up the closest one.
    float min_dist = std::numeric_limits<float>::max();
    cluster_id_t min_id = 0;
    for (size_t i = 0; i < k; i++)
    {
        if (dists[i] < min_dist)
        {
            min_dist = dists[i];
            min_id = i; 
        }
    }

    return std::make_pair(min_id, min_dist);
}


template<typename vector_dimension_t>
std::pair<std::vector<std::vector<vector_dimension_t>>, std::vector<cluster_id_t>> Partition<vector_dimension_t>::KMeans(
    const std::vector<std::vector<vector_dimension_t>>& obs, 
    size_t k, 
    size_t iter, 
    const std::string & minit // Method initial
)
{
    size_t N = obs.size();
    size_t D = obs[0].size();

    // Initialize centroids based on minit
    std::vector<std::vector<vector_dimension_t>> centroids(k, std::vector<vector_dimension_t>(D));
    if (minit == "points")
    {
        std::default_random_engine rd;
        // std::mt19937 gen(rd());
        std::uniform_int_distribution<vector_id_t> dist(0, N - 1);

        std::vector<vector_id_t> initial_indices(k);
        for (size_t i = 0; i < k; i++) {
            vector_id_t idx = dist(rd);
            initial_indices[i] = idx;
            centroids[i] = obs[idx];
        }
    }
    else {
        std::cerr << "Unknown clustering initial method: " << minit << "\n";
        exit(1);
    }
    
    // Perform k-means iterations
    std::vector<cluster_id_t> labels(N);
    for (size_t iter_count = 0; iter_count < iter; iter_count++) {
        // Assign each observation to the nearest centroid    
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            float min_dist;
            std::tie(labels[i], min_dist) = NearestCenter(obs[i], centroids);
        }
        // Update centroids based on assigned observations
        #pragma omp parallel for
        for (size_t j = 0; j < k; j++) {
            std::vector<float> sum(D, 0.0);
            size_t count = 0;
            for (size_t i = 0; i < N; i++) {
                if (labels[i] == j) {
                    std::transform(sum.begin(), sum.end(), obs[i].begin(), sum.begin(), std::plus<float>());
                    count++;
                }
            }
            // To match the centroids and the labels, 
            // no update should be done in the last iter. 
            if (iter_count < iter - 1 && count > 0) {
                std::transform(sum.begin(), sum.end(), centroids[j].begin(), [count](float val) { return val / count; });
            }
        }
    }
    return std::make_pair(centroids, labels);

}


};