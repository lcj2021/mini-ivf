#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
// #include <unordered_set>
#include <numeric>

#include "index_ivfpq.h"
#include "product_quantizer.h"

int main() {
    // dimension of the vectors to index
    int D = 64;
    // size of the database we plan to index
    size_t nb = 100'000;
    // make a set of nt training vectors in the unit cube (could be the database)
    size_t nt = 15'000;
    // size of the queries we plan to search
    int nq = 200;

    // a reasonable number of centroids to index nb vectors
    int ncentroids = 25;
    // nprobe
    int nprobe = 10;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // training
    std::vector<std::vector<float>> trainvecs(nt, std::vector<float>(D));
    for (size_t i = 0; i < nt; ++i) {
        for (size_t j = 0; j < D; ++j) {
            trainvecs[i][j] = distrib(rng);
        }
    }
    ProductQuantizer PQ(16, 8);
    auto & codewords = PQ.fit(trainvecs);

    // populating (adding) the database
    std::vector<std::vector<float>> database(nb, std::vector<float>(D));
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < D; ++j) {
            database[i][j] = distrib(rng);
        }
    }
    Toy::IndexIVFPQ index(codewords, ncentroids, true);
    index.AddCodes(PQ.encode(database), false);
    index.Reconfigure(ncentroids, 5);

    // searching the database
    std::vector<std::vector<float>> queries(nq, std::vector<float>(D));
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < D; ++j) {
            queries[i][j] = distrib(rng);
        }
    }

    std::vector<size_t> gt_nnid(nq);
    std::vector<float> gt_dist(nq);
    for (size_t i = 0; i < nq; ++i) {
        size_t cand_id = -1;
        float cand_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < nb; ++j) {
            float dist = PQ.compute_distance(queries[i], database[j]);
            if (cand_dist > dist) {
                cand_dist = dist;
                cand_id = j;
            }
        }
        assert(cand_id != -1);
        gt_dist[i] = cand_dist;
        gt_nnid[i] = cand_id;
        // std::cout << gt_nnid[i] << ' ' << gt_dist[i] << '\n';
    }

    puts("Index find kNN!");

    int k = 1;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));

    for (size_t i = 0; i < nq; ++i) {
        tie(nnid[i], dist[i]) = index.query(queries[i], std::vector<int>{}, k, nb, nprobe);
    }
        
    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i)
            if (nnid[q][i] == gt_nnid[q])
                n_ok++;
    }
    std::cout << (double)n_ok / nq << '\n';

    return 0;
}