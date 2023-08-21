#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.h"
#include "index_ivf.h"
#include "quantizer.h"
#include "util.h"

size_t D;              // dimension of the vectors to index
size_t nb;       // size of the database we plan to index
size_t nt;         // make a set of nt training vectors in the unit cube (could be the database)
int ncentroids = 1;
int nprobe = ncentroids;

int main() {
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, "../../dataset/sift/sift_base.fvecs");

    const auto& query = database;

    Toy::IVFConfig cfg(nb, D, nprobe, nb, 
                    ncentroids,
                    1, 
                    D);
    Toy::IndexIVF index(cfg, true, false);
    index.train(database, 123, true);
    index.populate(database);

    puts("Index find kNN!");
    // Recall@k
    int k = 100;
    int nq = 1000'000;
    std::vector<int> nnid(nq * k);
    std::vector<float>  dist(nq * k);
    Timer timer_query;
    timer_query.start();
#pragma omp parallel for
    for (size_t q = 0; q < nq; ++q) {
        const auto& [id, d] = index.query(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            std::vector<int>(), k, nb);
        
        size_t start_pos = q * k;
        for (size_t i = 0; i < k; ++i) {
            nnid[start_pos + i] = id[i];
            dist[start_pos + i] = d[i];
        }
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    write_to_file_binary(nnid, {nq, k}, "../../dataset/sift/sift_train_groundtruth.ivecs");
    write_to_file_binary(dist, {nq, k}, "../../dataset/sift/sift_train_distance.fvecs");
    return 0;
}