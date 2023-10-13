#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.h"
#include "index_ivfpq.h"
#include "quantizer.h"
#include "util.h"

size_t D;              // dimension of the vectors to index
size_t nb;       // size of the database we plan to index
size_t nt;         // make a set of nt training vectors in the unit cube (could be the database)
size_t mp = 128;
size_t nq = 2'000;
int ncentroids = 1'000;

int main(int argc, char* argv[]) {
    assert(argc == 2);
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, "/RF/dataset/sift/sift_base.fvecs");

    // auto& query = database;
    std::vector<float> query;
    load_from_file_binary(query, "/RF/dataset/sift/sift_query.fvecs");

    std::vector<int> gt;
    // load_from_file_binary(gt, "/RF/dataset/sift/sift_train_groundtruth.ivecs");
    load_from_file_binary(gt, "/RF/dataset/sift/sift_query_groundtruth.ivecs");

    int nprobe = std::atoi(argv[1]);

    Toy::IVFPQConfig cfg(nb, D, nprobe, nb, 
                    ncentroids, 256, 
                    1, mp, 
                    D, D / mp);
    Toy::IndexIVFPQ index(cfg, nq, true, false);
    std::string index_path = "/RF/index/sift/sift1m_pq" + std::to_string(mp)
                        + "_kc" + std::to_string(ncentroids);
    // index.train(database, 123, true);
    // index.write(index_path);
    index.load(index_path);
    index.populate(database);

    // std::vector<int> posting_lists_size(ncentroids);
    // for (size_t i = 0; i < ncentroids; i++) {
    //     posting_lists_size[i] = index.posting_lists_[i].size();
    // }
    // write_to_file_binary(posting_lists_size, {ncentroids, 1}, 
    //     "../../dataset/sift/sift1m_pq128_10k_kc1000_seg20_sample/posting_lists_size.ivecs");

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
        index.query_baseline(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            nnid[q], dist[q], searched_cnt, 
             k, nb, q);
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
    printf("PQ%lu, kc%d, W%d\n", mp, ncentroids, nprobe);

    return 0;
}