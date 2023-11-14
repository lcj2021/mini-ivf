#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <numeric>

#include "index_ivf.hpp"
#include "quantizer.hpp"
#include "util.hpp"


size_t D = 128;              // dimension of the vectors to index
size_t nb = 1'000'000;       // size of the database we plan to index
size_t nt = 1'000'000;         // make a set of nt training vectors in the unit cube (could be the database)
// size_t nb = 10'000;       // size of the database we plan to index
// size_t nt = 1'000;         // make a set of nt training vectors in the unit cube (could be the database)
size_t nq = 1'000;
int ncentroids = 4096;
int nprobe = 1'100;

toy::IVFConfig cfg(nb, D, nb / 50, 
    ncentroids, 1, D,
    "", ""
);

int main() {
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // training
    std::vector<std::vector<float>> trainvecs(nt, std::vector<float>(D));
    std::vector<float> trainvecs_flat(nt * D);
    for (size_t i = 0; i < nt; ++i) {
        for (size_t j = 0; j < D; ++j) {
            trainvecs[i][j] = distrib(rng);
            trainvecs_flat[i * D + j] = trainvecs[i][j];
        }
    }

    // populating (adding) the database
    std::vector<std::vector<float>> database(nb, std::vector<float>(D));
    std::vector<float> database_flat(nb * D);
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < D; ++j) {
            database[i][j] = distrib(rng);
            database_flat[i * D + j] = database[i][j];
        }
    }
    toy::IndexIVF<float> index(cfg, nq, true);
    index.Train(database_flat, 123, false);
    index.Populate(database_flat);
    // index.Reconfigure(ncentroids, 5);

    // searching the database
    std::vector<float> query(nq * D);
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < D; ++j) {
            query[i * D + j] = distrib(rng);
        }
    }

    std::vector<size_t> gt_nnid(nq);
    std::vector<float> gt_dist(nq);
    for (size_t i = 0; i < nq; ++i) {
        size_t cand_id = -1;
        float cand_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < nb; ++j) {
            float dist = fvec_L2sqr(query.data() + i * D, database[j].data(), D);
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

    // Recall@k
    int k = 10;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    Timer timer_query;
    timer_query.Start();
    size_t total_searched_cnt = 0;

    // #pragma omp parallel for reduction(+ : total_searched_cnt)
    for (size_t q = 0; q < nq; ++q) {
        size_t searched_cnt;
        index.QueryBaseline(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            nnid[q], dist[q], searched_cnt, 
            k, nb, q, nprobe
        );
        total_searched_cnt += searched_cnt;
    }
    timer_query.Stop();
    std::cout << timer_query.GetTime() << " seconds.\n";
        
    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i)
            if (nnid[q][i] == gt_nnid[q])
                n_ok++;
    }
    std::cout << (double)n_ok / nq << '\n';

    return 0;
}