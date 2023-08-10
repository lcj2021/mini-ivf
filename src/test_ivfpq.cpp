#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <numeric>

#include "index_ivfpq.h"
#include "product_quantizer.h"
#include "util.h"

int main() {
    // dimension of the vectors to index
    size_t D = 64;
    // size of the database we plan to index
    size_t nb = 1000'000;
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
    ProductQuantizer PQ(D, 16, 8);
    auto & codewords = PQ.fit(trainvecs, 10);

    // populating (adding) the database
    std::vector<std::vector<float>> database(nb, std::vector<float>(D));
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < D; ++j) {
            database[i][j] = distrib(rng);
        }
    }
    Toy::IndexIVFPQ index(codewords, D, ncentroids, 16, 8, true);
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
            float dist = Toy::fvec_L2sqr(queries[i].data(), database[j].data(), D);
            if (cand_dist > dist) {
                cand_dist = dist;
                cand_id = j;
            }
        }
        assert(cand_id != -1);
        gt_dist[i] = cand_dist;
        gt_nnid[i] = cand_id;
    }

    puts("Index find kNN!");

    // Recall@1
    int k = 1;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));

    Timer timer_query;
    timer_query.start();
    for (size_t i = 0; i < nq; ++i) {
        tie(nnid[i], dist[i]) = index.query(queries[i], std::vector<int>{}, k, nb, nprobe);
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";
        
    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i)
            if (nnid[q][i] == gt_nnid[q])
                n_ok++;
    }
    std::cout << (double)n_ok / nq << '\n';

    return 0;
}