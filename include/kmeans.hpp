#ifndef INCLUDE_KMEANS_HPP
#define INCLUDE_KMEANS_HPP

#include <index.hpp>

#include <tuple>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <cassert>

#include <distance.hpp>

namespace index {

template <typename vector_dimension_t> class KMeans {
public:
    // Linear search by L2 Distance computation. Return the best one (id, distance)
    static std::pair<cluster_id_t, distance_t> NearestCenter(
        const std::vector<vector_dimension_t> & query, 
        const std::vector<std::vector<vector_dimension_t>> & centers
    );

    // kmeans Lloyd implementation
    static std::tuple<std::vector<std::vector<distance_t>>, std::vector<cluster_id_t>> 
    KMeans(
        const std::vector<std::vector<vector_dimension_t>>& obs, 
        size_t k, 
        size_t iter, 
        const std::string& minit
    );
};


/**
 * Template class declarations.
*/
template class KMeans<uint8_t>;
template class KMeans<uint16_t>;
template class KMeans<uint32_t>;
template class KMeans<uint64_t>;
template class KMeans<float>;
template class KMeans<double>;

};

#endif