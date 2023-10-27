#ifndef INCLUDE_KMEANS_HPP
#define INCLUDE_KMEANS_HPP

#include <tuple>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <cassert>

#include "distance.hpp"

// Linear search by L2 Distance computation. Return the best one (id, distance)
std::pair<size_t, float> 
nearest_center(const std::vector<float>& query, const std::vector<std::vector<float>>& centers);

// kmeans Lloyd implementation
std::tuple<std::vector<std::vector<float>>, std::vector<int>> 
KMeans(const std::vector<std::vector<float>>& obs, int k, int iter, const std::string& minit);


#endif