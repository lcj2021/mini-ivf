#ifndef INCLUDE_PARTITION_HPP
#define INCLUDE_PARTITION_HPP

#include <index.hpp>

#include <tuple>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <cassert>
#include <distance.hpp>

namespace index {

template <typename vector_dimension_t> class Partition {
public:
    // Linear search by L2 Distance computation. Return the best one (id, distance)
    static std::pair<cluster_id_t, float> NearestCenter(
        const std::vector<vector_dimension_t> & query, 
        const std::vector<std::vector<vector_dimension_t>> & centers
    );

    // kmeans Lloyd implementation
    static std::tuple<std::vector<std::vector<float>>, std::vector<cluster_id_t>> 
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
template class Partition<uint8_t>;
template class Partition<float>;


};

#endif