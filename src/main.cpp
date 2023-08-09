#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <unordered_set>

#include "rii.h"
#include "product_quantizer.h"

int main() {
    // dimension of the vectors to index
    int d = 64;

    // size of the database we plan to index
    size_t nb = 100'000;

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 15'000;

    // a reasonable number of centroids to index nb vectors
    int ncentroids = 25;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // training
    std::vector<std::vector<float>> trainvecs(nt, std::vector<float>(d));
    for (size_t i = 0; i < nt; ++i) {
        for (size_t j = 0; j < d; ++j) {
            trainvecs[i][j] = distrib(rng);
        }
    }
    ProductQuantizer PQ(16, 8);
    auto & codewords = PQ.fit(trainvecs);

    // populating the database

    std::vector<std::vector<float>> database(nb, std::vector<float>(d));
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < d; ++j) {
            database[i][j] = distrib(rng);
        }
    }
    rii::RiiCpp index(codewords, ncentroids, true);
    index.AddCodes(PQ.encode(database), false);
    index.Reconfigure(ncentroids, 5);
    // index.add(nb, database.data());
    // index_gt.add(nb, database.data());

    int nq = 100;
    int n_ok;

    // searching the database
    std::vector<std::vector<float>> queries(nq, std::vector<float>(d));
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < d; ++j) {
            queries[i][j] = distrib(rng);
        }
    }

    std::vector<size_t> gt_nns(nq);
    std::vector<float> gt_dis(nq);
    for (size_t i = 0; i < nq; ++i) {
        size_t cand_id = -1;
        float cand_dist = MAXFLOAT;
        for (int j = 0; j < nb; ++j) {
            float dist = PQ.compute_distance(queries[i], database[j]);
            if (cand_dist > dist) {
                cand_dist = dist;
                cand_id = j;
            }
        }
        assert(cand_id != -1);
        gt_dis[i] = cand_dist;
        gt_nns[i] = cand_id;
        // std::cout << gt_nns[i] << ' ' << gt_dis[i] << '\n';
    }

    puts("Index find kNN!");

    int k = 5;
    std::vector<std::vector<size_t>> nns(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dis(nq, std::vector<float>(k));

    for (size_t i = 0; i < nq; ++i) {
        tie(nns[i], dis[i]) = index.QueryIvf(queries[i], std::vector<int>{}, k, nb, ncentroids);
    }
        

    n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i)
            if (nns[q][i] == gt_nns[q])
                n_ok++;
    }
    std::cout << (double)n_ok / nq << '\n';

    return 0;
}