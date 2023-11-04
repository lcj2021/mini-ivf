// #include "kmeans.hpp"

// #include <algorithm>


// // Linear search by L2 Distance computation. Return the best one (id, distance)
// template <typename T>
// std::pair<size_t, float> 
// NearestCenter(const std::vector<T>& query, const std::vector<std::vector<float>>& centers) 
// {
//     std::vector<float> dists(centers.size());
//     size_t K_ = centers.size(), Ds_ = centers[0].size();

//     /**
//      * @bug NO OMP here!
//     */
//     // #pragma omp parallel for
//     for (size_t i = 0; i < K_; ++i) {
//         dists[i] = fvec_L2sqr(query.data(), centers[i].data(), Ds_);
//     }

//     // Just pick up the closest one
//     float min_dist = std::numeric_limits<float>::max();
//     int min_i = -1;
//     for (size_t i = 0; i < K_; ++i) {
//         if (dists[i] < min_dist) {
//             min_i = static_cast<int>(i);
//             min_dist = dists[i];
//         }
//     }
//     assert(min_i != -1);

//     return std::pair<size_t, float>((size_t)min_i, min_dist);
// }

// // kmeans Lloyd implementation
// template <typename T>
// std::tuple<std::vector<std::vector<float>>, std::vector<int>> 
// KMeans(const std::vector<std::vector<T>>& obs, int k, int iter, const std::string& minit) {
//     int N = obs.size();
//     int D = obs[0].size();

//     // Initialize centroids based on minit
//     std::vector<std::vector<float>> centroids(k, std::vector<float>(D, 0.0));
//     if (minit == "points") {
//         std::default_random_engine rd;
//         // std::mt19937 gen(rd());
//         std::uniform_int_distribution<int> dist(0, N - 1);

//         std::vector<int> initial_indices(k);
//         for (int i = 0; i < k; ++i) {
//             int idx = dist(rd);
//             initial_indices[i] = idx;
//             centroids[i] = obs[idx];
//         }
//     }

//     // Perform k-means iterations
//     std::vector<int> labels(N);
//     for (int iter_count = 0; iter_count < iter; ++iter_count) {
//         // std::cout << iter_count << '\n';
//         // Assign each observation to the nearest centroid
//         double errors_sum = 0.0;
        
//         #pragma omp parallel for
//         for (int i = 0; i < N; ++i) {
//             float min_dist;
//             std::tie(labels[i], min_dist) = NearestCenter(obs[i], centroids);
//         }

//         // Update centroids based on assigned observations
//         #pragma omp parallel for
//         for (int j = 0; j < k; ++j) {
//             std::vector<float> sum(D, 0.0);
//             int count = 0;
//             for (int i = 0; i < N; ++i) {
//                 if (labels[i] == j) {
//                     std::transform(sum.begin(), sum.end(), obs[i].begin(), sum.begin(), std::plus<float>());
//                     ++count;
//                 }
//             }
//             // To match the centroids and the labels, 
//             // no update should be done in the last iter. 
//             if (iter_count != iter - 1 && count > 0) {
//             // if (count > 0) {
//                 std::transform(sum.begin(), sum.end(), centroids[j].begin(), [count](float val) { return val / count; });
//             }
//         }
//     }

//     return {centroids, labels};
// }


// template <> std::pair<size_t, float> 
// NearestCenter<uint8_t>(const std::vector<uint8_t>& query, const std::vector<std::vector<float>>& centers);
// template <> std::pair<size_t, float> 
// NearestCenter<float>(const std::vector<float>& query, const std::vector<std::vector<float>>& centers);

// // template <> std::tuple<std::vector<std::vector<float>>, std::vector<int>> 
// // KMeans<uint8_t>(const std::vector<std::vector<uint8_t>>& obs, int k, int iter, const std::string& minit);
// // template <> std::tuple<std::vector<std::vector<float>>, std::vector<int>> 
// // KMeans<float>(const std::vector<std::vector<float>>& obs, int k, int iter, const std::string& minit);