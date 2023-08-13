#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "hdf5_io.h"
#include "index_rii.h"
#include "quantizer.h"
#include "util.h"

int main() {
    // dimension of the vectors to index
    // size of the database we plan to index
    // make a set of nt training vectors in the unit cube (could be the database)
    // size of the queries we plan to search

    std::vector<float> database;
    auto [nb, D] = load_from_file(database, "../../dataset/sift-128-euclidean.hdf5", "train");

    std::vector<float> query;
    auto [nq, _] = load_from_file(query, "../../dataset/sift-128-euclidean.hdf5", "test");

    std::vector<int> gt;
    load_from_file(gt, "../../dataset/sift-128-euclidean.hdf5", "neighbors");

    size_t M = 64, nbits = 8;

    Quantizer::Quantizer CQ(D, nb, M, 1LL << nbits, 5, true);
    CQ.fit(database, 5, 123);
    const auto& codewords_cq = CQ.get_centroids();

    // a reasonable number of centroids to index nb vectors
    int ncentroids = 100;
    // nprobe
    int nprobe = 100;
    // Toy::IndexRII index(codewords, D, ncentroids, M, nbits, true, false);
    // index.AddCodes(PQ.encode(database), false);
    Toy::IndexRII index(codewords_cq, D, ncentroids, M, nbits, true, true);
    const auto& encodewords = CQ.encode(database);
    index.AddCodes(encodewords, false);
    index.Reconfigure(ncentroids, 5);

    puts("Index find kNN!");
    // Recall@k
    int k = 100;
    nq = 10'000;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    Timer timer_query;
    timer_query.start();
    for (size_t q = 0; q < nq; ++q) {
        tie(nnid[q], dist[q]) = index.query(std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
        std::vector<int>(gt.begin() + q * 100, gt.begin() + q * 100 + k), k, nb, nprobe);
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<int> S(gt.begin() + q * 100, gt.begin() + q * 100 + k);
        for (int i = 0; i < k; ++i)
            if (S.count(nnid[q][i]))
                n_ok++;
    }
    std::cout << (double)n_ok / (nq * k) << '\n';

    return 0;
}