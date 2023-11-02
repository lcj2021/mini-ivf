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
size_t mp = 128;
size_t nq = 5;
int ncentroids = 1000;
int nprobe = ncentroids;

std::string index_path = "/RF/index/sift/sift1m_pq" + std::to_string(mp)
                + "_kc" + std::to_string(ncentroids);
std::string db_path = "/RF/dataset/sift";


int main() {
    std::vector<float> database;
    std::tie(nb, D) = LoadFromFileBinary<float>(database, db_path + "/sift_base.fvecs");

    // const auto& query = database;
    std::vector<float> query;
    LoadFromFileBinary<float>(query, db_path + "/sift_query.fvecs");

    std::vector<int> gt;
    // LoadFromFileBinary(gt, db_path + "/sift_train_groundtruth.ivecs");
    LoadFromFileBinary<int>(gt, db_path + "/sift_query_groundtruth.ivecs");

    toy::IVFPQConfig cfg(
        nb, D, nb, 
        ncentroids, 256, 
        1, mp, 
        D, D / mp, 
        index_path, db_path
    );
    toy::IndexIVFPQ index(cfg, nq, true);
    // index.Train(database, 123, true);
    // index.WriteIndex(index_path);
    // index.LoadIndex(index_path);
    // index.Populate(database);
    std::vector<uint32_t> book{0, 2, 4, 67, 21, 123, 321, 234, 567, 764, 232};
    index.load_from_book(book, db_path + "/sift1m_pq128_kc1000_cluster");

    std::iota(book.begin(), book.end(), (uint32_t)0);
    index.load_from_book(book, db_path + "/sift1m_pq128_kc1000_cluster");


    index.LoadPqCodebook(db_path + "/sift1m_pq128_kc1000_cluster");
    index.LoadCqCodebook(db_path + "/sift1m_pq128_kc1000_cluster");

    query.resize(nq * D);
    auto nested_queries = nest_2d(query, {nq, D});
    std::vector<std::vector<uint32_t>> topw;
    index.TopWId(50, nested_queries, topw, 16);
    for (size_t n = 0; n < nq; ++n) {
        std::cerr << "Query " << n << " :\n";
        for (const auto& id : topw[n]) {
            std::cerr << id << ", ";
        }
        std::cerr << std::endl;
    }

    // index.SetClusterVectorPath(db_path + "/sift1m_pq128_kc1000_cluster");
    // index.SetClusterIdPath(db_path + "/sift1m_pq128_kc1000_cluster");
    index.Finalize();

    return 0;
}