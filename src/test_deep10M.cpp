#include <random>
#include <iostream>
#include <numeric>

#include "hdf5_io.h"
#include "index_ivfpq.h"
#include "product_quantizer.h"
#include "util.h"

int main() {
    // dimension of the vectors to index
    // size of the database we plan to index
    // make a set of nt training vectors in the unit cube (could be the database)
    // size of the queries we plan to search

    std::vector<float> database;
    auto [nb, D] = load_from_file(database, "../../dataset/deep-image-96-angular.hdf5", "train");

    std::vector<float> query;
    auto [nq, _] = load_from_file(query, "../../dataset/deep-image-96-angular.hdf5", "test");

    std::vector<int> gt;
    load_from_file(gt, "../../dataset/deep-image-96-angular.hdf5", "neighbors");

    ProductQuantizer PQ(D, 32, 8);
    auto & codewords = PQ.fit(database, 1);

    // a reasonable number of centroids to index nb vectors
    int ncentroids = 25;
    // nprobe
    int nprobe = 10;
    Toy::IndexIVFPQ index(codewords, D, ncentroids, 32, 8, true);
    index.AddCodes(PQ.encode(database), false);
    index.Reconfigure(ncentroids, 5);

    puts("Index find kNN!");
    // Recall@1
    int k = 1;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    Timer timer_query;
    timer_query.start();
    for (size_t i = 0; i < nq; ++i) {
        // tie(nnid[i], dist[i]) = index.query(queries[i], std::vector<int>{}, k, nb, nprobe);
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    return 0;
}