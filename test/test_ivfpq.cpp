#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <numeric>
#include <distance.hpp>
#include <cassert>
#include <ivf/index_ivfpq.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>


size_t D = 128;                 // dimension of the vectors to index
size_t nb = 100'000;            // size of the database we plan to index
size_t nt = 15'000;             // make a set of nt training vectors in the unit cube (could be the database)
size_t kc = 1000;
size_t kp = 256;
size_t mp = 64;
size_t nq = 2'000;              // size of the query we plan to search
size_t k = 100;
size_t L = nb;
size_t nprobe = 20;


int main() {
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // training
    std::vector<float> trainvecs_flat(nt * D);
    for (size_t i = 0; i < nt; ++i) {
        for (size_t j = 0; j < D; ++j) {
            trainvecs_flat[i * D + j] = distrib(rng);
        }
    }

    // populating (adding) the database
    std::vector<float> database_flat(nb * D);
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < D; ++j) {
            database_flat[i * D + j] = distrib(rng);
        }
    }

    // searching the database
    std::vector<float> query(nq * D);
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < D; ++j) {
            query[i * D + j] = distrib(rng);
        }
    }

    index::ivf::IndexIVFPQ<float> ivfpq(
        nb, D, L, kc, kp, mp, 
        "../data/index/test_ivfpq/", "../data/db/test_ivfpq/", "TEST-IVFPQ",
        index::IndexStatus::LOCAL
    );

    std::cout << "Hello world from " << ivfpq.GetName() << std::endl;

    ivfpq.Train(trainvecs_flat);

    ivfpq.Populate(database_flat);

    puts("Gt searching...");

    std::vector<index::cluster_id_t> gt_nnid(nq);
    std::vector<float> gt_dist(nq);
    for (size_t i = 0; i < nq; ++i) {
        size_t cand_id = -1;
        float cand_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < nb; ++j) {
            float dist = vec_L2sqr(query.data() + i * D, database_flat.data() + j * D, D);
            if (cand_dist > dist) {
                cand_dist = dist;
                cand_id = j;
            }
        }
        assert(cand_id != -1);
        gt_dist[i] = cand_dist;
        gt_nnid[i] = cand_id;
    }

    puts("Test IVFPQ searching...");

    // Recall@k
    std::vector<std::vector<index::cluster_id_t>> nnid(nq);
    std::vector<std::vector<float>> dists(nq);
    auto query_nest = utils::Resize<float>::Nest(query, nq, D);
    utils::STimer tq;

    tq.Start();
    std::vector<std::vector<index::cluster_id_t>> topw_books;
    ivfpq.TopWID(nprobe, query_nest, topw_books);
    ivfpq.TopKID(k, query_nest, topw_books, nnid, dists);
    tq.Stop();

    std::cout << tq.GetTime() << " seconds.\n";
        
    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < k; ++i)
            if (nnid[q][i] == gt_nnid[q])
                n_ok++;
    }
    std::cout << "Recall@" << k << ": " << (double)n_ok / nq << '\n';

    return 0;
}