#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.hpp"
#include "index_ivfpq.hpp"
#include "quantizer.hpp"
#include "util.hpp"

size_t D;              // dimension of the vectors to index
size_t nb;       // size of the database we plan to index
size_t nt = 200'000;         // make a set of nt training vectors in the unit cube (could be the database)
size_t mp = 960;
size_t nq = 1'000;
int ncentroids = 1'000;

std::string index_path = "/RF/index/gist1m/gist1m_pq" + std::to_string(mp)
                    + "_kc" + std::to_string(ncentroids);
std::string db_path = "/RF/dataset/gist1m";

int main(int argc, char* argv[]) {
    assert(argc == 2);
    modify_path(index_path);
    modify_path(db_path);
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary<float>(database, db_path + "base.fvecs");

    // auto& query = database;
    std::vector<float> query;
    load_from_file_binary<float>(query, db_path + "query.fvecs");

    std::vector<int> gt;
    // load_from_file_binary(gt, db_path + "train_groundtruth.ivecs");
    load_from_file_binary<int>(gt, db_path + "query_groundtruth.ivecs");

    int nprobe = std::atoi(argv[1]);

    Toy::IVFPQConfig cfg(
        nb, D, nb, 
        ncentroids, 256, 
        1, mp, 
        D, D / mp, 
        index_path, db_path
    );
    Toy::IndexIVFPQ index(cfg, nq, true);
    // index.train(database, 123, true);
    // index.write_index(index_path);
    index.load_index(index_path);
    std::vector<uint32_t> book(ncentroids);
    std::iota(book.begin(), book.end(), (uint32_t)0);
    index.load_from_book(book, db_path + "/gist1m_pq" + std::to_string(mp) + "_kc1000_cluster");
    // index.populate(database);

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
        index.query_baseline(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            nnid[q], dist[q], searched_cnt, 
             k, nb, q, nprobe
        );
        total_searched_cnt += searched_cnt;
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    index.finalize();

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