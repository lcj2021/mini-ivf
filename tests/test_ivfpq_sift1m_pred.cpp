#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.h"
#include "index_ivfpq.h"
#include "quantizer.h"
#include "util.h"

size_t D, d_pred;              // dimension of the vectors to index
size_t nb;       // size of the database we plan to index
size_t nt;         // make a set of nt training vectors in the unit cube (could be the database)
size_t n_pred;
size_t mp = 128;
size_t nq = 2'000;
size_t segs = 20;
int ncentroids = 100;
int nprobe = ncentroids;

int main() {
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, "/RF/dataset/sift/sift_base.fvecs");

    // auto& query = database;
    std::vector<float> query;
    load_from_file_binary(query, "/RF/dataset/sift/sift_query.fvecs");

    std::vector<int> gt;
    // load_from_file_binary(gt, "/RF/dataset/sift/sift_train_groundtruth.ivecs");
    load_from_file_binary(gt, "/RF/dataset/sift/sift_query_groundtruth.ivecs");

    std::vector<int> pred_radius;
    std::tie(n_pred, d_pred) = load_from_file_binary(pred_radius, 
                "/RF/dataset/sift/sift1m_pq128_10k_kc100_seg20/pred.ivecs");

    assert(d_pred == ncentroids);

    Toy::IVFPQConfig cfg(nb, D, nprobe, nb, 
                    ncentroids, 256, 
                    1, mp, 
                    D, D / mp, segs);
    Toy::IndexIVFPQ index(cfg, nq, true, true);
    // index.train(database, 123, true);
    index.load("/RF/index/sift/sift1m_pq128_kc100");
    index.populate(database);

    puts("Index find kNN!");
    // Recall@k
    int k = 100;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    size_t searched_cnt = 0;
    Timer timer_query;
    timer_query.start();
// #pragma omp parallel for
    for (size_t q = 0; q < nq; ++q) {
        const auto& res = index.query_pred(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            std::vector<int>(pred_radius.begin() + q * d_pred, pred_radius.begin() + q * d_pred + k), 
            k, nb, q);
        tie(nnid[q], dist[q]) = res.first;
        searched_cnt += res.second;
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
    std::cout << "avg_searched_cnt: " << (double)searched_cnt / nq << '\n';

    return 0;
}