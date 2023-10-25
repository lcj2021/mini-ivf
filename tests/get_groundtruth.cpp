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
size_t nt = 200'000;          // make a set of nt training vectors in the unit cube (could be the database)
size_t nq = 2'000;
int ncentroids = 1;
int nprobe = ncentroids;

std::string suffix = "nt" + to_string_with_units(nt) 
                    + "_kc" + std::to_string(ncentroids);

std::string index_path = std::string("/home/anns/index/sift1m/")
                    + suffix;
std::string db_path = "/RF/dataset/sift1m";

int main() {
    std::vector<float> database;
    std::tie(nb, D) = load_from_file_binary(database, db_path + "/base.fvecs");

    const auto& query = database;

    Toy::IVFConfig cfg(
        nb, D, nb, 
        ncentroids, 1, D, 
        index_path, db_path
    );
    Toy::IndexIVF index(cfg, nq, true);
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
        index.query_baseline(
            std::vector<float>(query.begin() + q * D, query.begin() + (q + 1) * D), 
            nnid[q], dist[q], searched_cnt, k, nb, q, nprobe
        );
        total_searched_cnt += searched_cnt;
        
        // size_t start_pos = q * k;
        // for (size_t i = 0; i < k; ++i) {
        //     nnid[start_pos + i] = id[i];
        //     dist[start_pos + i] = d[i];
        // }
    }
    timer_query.stop();
    std::cout << timer_query.get_time() << " seconds.\n";

    write_to_file_binary(nnid, {nq, k}, db_path + "/train_groundtruth.ivecs");
    write_to_file_binary(dist, {nq, k}, db_path + "/train_distance.fvecs");
    return 0;
}