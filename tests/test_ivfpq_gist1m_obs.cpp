#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.hpp"
#include "index_ivfpq.hpp"
#include "quantizer.hpp"
#include "util.hpp"

size_t D;           // dimension of the vectors to index
size_t nb;          // size of the database we plan to index
size_t nt;          // make a set of nt training vectors in the unit cube (could be the database)
size_t mp = 4;
size_t nq = 1'0;
int ncentroids = 100;
int nprobe = ncentroids;

std::string index_path = "/RF/index/gist1m/gist1m_pq" + std::to_string(mp)
                    + "_kc" + std::to_string(ncentroids);
std::string db_path = "/RF/dataset/gist1m";

int main() {
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, db_path + "base.fvecs");

    // const auto& query = database;
    std::vector<float> query;
    load_from_file_binary(query, db_path + "query.fvecs");

    std::vector<int> gt;
    // load_from_file_binary(gt, db_path + "train_groundtruth.ivecs");
    load_from_file_binary(gt, db_path + "query_groundtruth.ivecs");


    Toy::IVFPQConfig cfg(
        nb, D, nb, 
        ncentroids, 256, 
        1, mp, 
        D, D / mp,
        index_path, db_path
    );
    Toy::IndexIVFPQ index(cfg, nq, true);
    index.train(database, 123, true);
    index.write_index(index_path);
    index.load_index(index_path);
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
        index.query_obs(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            std::vector<int>(gt.begin() + q * 100, gt.begin() + q * 100 + k), 
            nnid[q], dist[q], searched_cnt, 
             k, nb, q);
        total_searched_cnt += searched_cnt;
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<int> S(gt.begin() + q * 100, gt.begin() + q * 100 + k);
        for (int i = 0; i < nnid[q].size(); ++i)
            if (S.count(nnid[q][i]))
                n_ok++;
    }
    std::cout << (double)n_ok / (nq * k) << '\n';

    index.show_statistics();

    return 0;
}