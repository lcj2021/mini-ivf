#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.hpp"
#include "index_ivf.hpp"
#include "quantizer.hpp"
#include "util.hpp"

size_t D;           // dimension of the vectors to index
size_t nb;          // size of the database we plan to index
size_t nt;          // make a set of nt training vectors in the unit cube (could be the database)
size_t mp = 64;
size_t nq = 1'000;
size_t segs = 20;
int ncentroids = 100;
int nprobe = ncentroids;

int main() {
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, "/RF/dataset/sift/sift_base.fvecs");

    const auto& query = database;
    // std::vector<float> query;
    // load_from_file_binary(query, "/RF/dataset/sift/sift_query.fvecs");

    std::vector<int> gt;
    load_from_file_binary(gt, "/RF/dataset/sift/sift_train_groundtruth.ivecs");
    // load_from_file_binary(gt, "/RF/dataset/sift/sift_query_groundtruth.ivecs");

    Toy::IVFConfig cfg(nb, D, nprobe, nb, 
                    ncentroids, 
                    1, 
                    D, segs);
    Toy::IndexIVF index(cfg, nq, true, true);
    index.train(database, 123, true);
    index.populate(database);

    puts("Index find kNN!");
    // Recall@k
    int k = 100;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    Timer timer_query;
    timer_query.start();
    size_t total_searched_cnt = 0;
    
    #pragma omp parallel for reduction(+ : total_searched_cnt)
    for (size_t q = 0; q < nq; ++q) {
        size_t searched_cnt;
        index.query_exhausted(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            std::vector<int>(gt.begin() + q * 100, gt.begin() + q * 100 + k), 
            nnid[q], dist[q], searched_cnt, 
             k, nb, q);
        total_searched_cnt += searched_cnt;
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    index.write_trainset("/RF/dataset/sift/sift1m_ivf_10k_kc100_seg20", 0);

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