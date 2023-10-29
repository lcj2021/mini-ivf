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
size_t mp = 128;
size_t nq = 1'000;
int ncentroids = 1'000;

std::string suffix = "nt" + ToStringWithUnits(nt) 
                    + "_pq" + std::to_string(mp)
                    + "_kc" + std::to_string(ncentroids);

std::string index_path = std::string("/dk/anns/index/sift10m/")
                    + suffix;
std::string db_path = "/dk/anns/dataset/sift10m";
std::string query_path = "/dk/anns/query/sift10m";

int main(int argc, char* argv[]) {
    assert(argc == 2);
    std::vector<float> database;
    std::tie(nb, D) = LoadFromFileBinary<float>(database, db_path + "/base.fvecs");

    std::vector<float> query;
    LoadFromFileBinary<float>(query, db_path + "/query.fvecs");

    std::vector<int> gt;
    LoadFromFileBinary<int>(gt, db_path + "/query_groundtruth.ivecs");

    int nprobe = std::atoi(argv[1]);

    toy::IVFPQConfig cfg(
        nb, D, nb, 
        ncentroids, 256, 
        1, mp, 
        D, D / mp, 
        index_path, db_path
    );
    toy::IndexIVFPQ index(cfg, nq, true);
    // index.LoadIndex(index_path);
    // std::vector<uint32_t> book(ncentroids);
    // std::iota(book.begin(), book.end(), (uint32_t)0);
    // index.load_from_book(book, "/home/anns/dataset/sift10m/" + suffix);
    index.Train(database, 123, true);
    // index.WriteIndex(index_path);
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
            nnid[q], dist[q], searched_cnt, 
             k, nb, q, nprobe
        );
        total_searched_cnt += searched_cnt;
    }
    timer_query.Stop();
    std::cout << timer_query.GetTime() << " seconds.\n";
    
    index.Finalize();

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