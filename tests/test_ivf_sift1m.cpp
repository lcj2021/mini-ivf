#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "hdf5_io.h"
#include "index_ivf.h"
#include "quantizer.h"
#include "util.h"

size_t D;              // dimension of the vectors to index
size_t nb;       // size of the database we plan to index
size_t nt;         // make a set of nt training vectors in the unit cube (could be the database)
int ncentroids = 1;
int nprobe = ncentroids;

int main() {
    // dimension of the vectors to index
    // size of the database we plan to index
    // make a set of nt training vectors in the unit cube (could be the database)
    // size of the queries we plan to search

    std::vector<float> database;
    std::tie(nb, D) = load_from_file_hdf5(database, "../../dataset/sift-128-euclidean.hdf5", "train");

    std::vector<float> query;
    auto [nq, _] = load_from_file_hdf5(query, "../../dataset/sift-128-euclidean.hdf5", "test");

    std::vector<int> gt;
    load_from_file_hdf5(gt, "../../dataset/sift-128-euclidean.hdf5", "neighbors");

    // std::vector<float> gt_d;
    // load_from_file_hdf5(gt_d, "../../dataset/sift-128-euclidean.hdf5", "distances");

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
    nq = 10'000;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    Timer timer_query;
    timer_query.start();
    for (size_t q = 0; q < nq; ++q) {
        tie(nnid[q], dist[q]) = index.query(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            std::vector<int>(gt.begin() + q * 100, gt.begin() + q * 100 + k), k, nb);
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