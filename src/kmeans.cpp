#include <kmeans.hpp>

namespace index {

template<typename vector_dimension_t>
static std::pair<cluster_id_t, distance_t> KMeans<vector_dimension_t>::NearestCenter(
    const std::vector<vector_dimension_t> & query, 
    const std::vector<std::vector<vector_dimension_t>> & centers
)
{
    size_t k = centers.size(), ds = centers[0].size();
    assert(k);
    std::vector<distance_t> dists(k);

    /// @bug No omp here.
    // #pragema omp parallel for
    for (size_t i = 0; i < K_; i++)
    {
        dists[i] = fvec_L2sqr(query.data(), centers[i].data(), ds);
    }

    // Just pick up the closest one.
    distance_t min_dist = std::numeric_limits<distance_t>::max();
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
static std::tuple<std::vector<std::vector<distance_t>>, std::vector<cluster_id_t>> KMeans<vector_dimension_t>::KMeans(
    const std::vector<std::vector<vector_dimension_t>>& obs, 
    size_t k, 
    size_t iter, 
    const std::string & minit
)
{
    size_t n = obs.size();
    size_t d = obs[0].size();

    // Initialize centroids based on minit
    std::vector<std::vector<vector_dimension_t>> centroids(k, std::vector<vector_dimension_t>(d));
    if (minit == "points")
    {
        std::default_random_engine rd;
        // std::mt19937 gen(rd());
        std::uniform_int_distribution<cluster_id_t> dist(0, n - 1);

        std::vector<cluster_id_t> initial_indices(k);
        for (size_t i = 0; i < k; i++) {
            cluster_id_t idx = dist(rd);
            initial_indices[i] = idx;
            centroids[i] = obs[idx];
        }
    }
    
}


};