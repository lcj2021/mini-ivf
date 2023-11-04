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

std::string suffix = "nt" + ToStringWithUnits(nt) 
                    + "_kc" + std::to_string(ncentroids);

std::string index_path = std::string("/home/anns/index/sift1m/")
                    + suffix;
std::string db_path = "/RF/dataset/sift1m";

int main() {
    std::vector<float> database;
    std::tie(nb, D) = LoadFromFileBinary<float>(database, db_path + "/base.fvecs");

    const auto& query = database;

    toy::IVFConfig cfg(
        nb, D, nb, 
        ncentroids, 1, D, 
        index_path, db_path
    );
    toy::IndexIVF<float> index(cfg, nq, true);
    index.Train(database, 123, nt);
    index.Populate(database);

    puts("Index find kNN!");
    // Recall@k
    int k = 100;
    std::vector<std::vector<size_t>> nnid(nq, std::vector<size_t>(k));
    std::vector<std::vector<float>> dist(nq, std::vector<float>(k));
    Timer timer_query;
    timer_query.Start();
    size_t total_searched_cnt = 0;

    #pragma omp parallel for reduction(+ : total_searched_cnt)
    for (size_t q = 0; q < nq; ++q) {
        size_t searched_cnt;
        index.QueryBaseline(
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
    timer_query.Stop();
    std::cout << timer_query.GetTime() << " seconds.\n";

    WriteToFileBinary(nnid, {nq, k}, db_path + "/train_groundtruth.ivecs");
    WriteToFileBinary(dist, {nq, k}, db_path + "/train_distance.fvecs");
    return 0;
}