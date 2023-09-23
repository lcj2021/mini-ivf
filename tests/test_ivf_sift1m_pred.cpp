#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.h"
#include "index_ivf.h"
#include "quantizer.h"
#include "util.h"

size_t D, d_pred;              // dimension of the vectors to index
size_t nb;       // size of the database we plan to index
size_t nt;         // make a set of nt training vectors in the unit cube (could be the database)
size_t n_pred;
size_t mp = 128;
size_t nq = 2'000;
size_t segs = 20;
int ncentroids = 1000;

int main(int argc, char* argv[]) {
    assert(argc == 4);
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, "/RF/dataset/sift/sift_base.fvecs");

    // auto& query = database;
    std::vector<float> query;
    load_from_file_binary(query, "/RF/dataset/sift/sift_query.fvecs");

    std::vector<int> gt;
    // load_from_file_binary(gt, "/RF/dataset/sift/sift_train_groundtruth.ivecs");
    load_from_file_binary(gt, "/RF/dataset/sift/sift_query_groundtruth.ivecs");

    std::vector<int> pred_radius;
    // "/RF/dataset/sift/sift1m_pq128_100k_kc100_seg20/pred.ivecs"
    std::string pred_path = std::string(argv[3]);
    std::tie(n_pred, d_pred) = load_from_file_binary(pred_radius, 
                pred_path);
    
    int cut = std::atoi(argv[1]);
    int nprobe = std::atoi(argv[2]);

    assert(d_pred == ncentroids);

    Toy::IVFConfig cfg(nb, D, nprobe, nb, 
                    ncentroids, 
                    1,  
                    D, segs);
    Toy::IndexIVF index(cfg, nq, true, false);
    index.train(database, 123, true);
    std::string index_path = "/RF/index/sift/sift1m_pq" + std::to_string(mp)
                        + "_kc" + std::to_string(ncentroids);
                        
    index.populate(database);

    puts("Index find kNN!");
    // Recall@k
    int k = 100;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    size_t total_searched_cnt = 0;
    Timer timer_query;
    timer_query.start();
// #pragma omp parallel for
    for (size_t q = 0; q < nq; ++q) {
        size_t searched_cnt;
        index.query_pred(
            std::vector<float>(query.data() + q * D, query.data() + (q + 1) * D), 
            std::vector<int>(pred_radius.data() + q * d_pred, pred_radius.data() + q * d_pred + k), 
            nnid[q], dist[q], searched_cnt, 
            cut, k, nb, q);
        total_searched_cnt += searched_cnt;
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
    std::cout << "Recall@" << k << ": " << (double)n_ok / (nq * k) << '\n';
    std::cout << "avg_searched_cnt: " << (double)total_searched_cnt / nq << '\n';
    printf("IVF, segs%lu, kc%d, W%d\n", segs, ncentroids, nprobe);
    printf("cut%d\n", cut);
    std::cout << "pred_path: " << pred_path << '\n';

    return 0;
}