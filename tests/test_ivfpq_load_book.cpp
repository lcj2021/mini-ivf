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
    std::tie(nb, D) = load_from_file_binary<float>(database, db_path + "/sift_base.fvecs");

    // const auto& query = database;
    std::vector<float> query;
    load_from_file_binary<float>(query, db_path + "/sift_query.fvecs");

    std::vector<int> gt;
    // load_from_file_binary(gt, db_path + "/sift_train_groundtruth.ivecs");
    load_from_file_binary<int>(gt, db_path + "/sift_query_groundtruth.ivecs");

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
    // index.load_index(index_path);
    // index.populate(database);
    std::vector<uint32_t> book{0, 2, 4, 67, 21, 123, 321, 234, 567, 764, 232};
    index.load_from_book(book, db_path + "/sift1m_pq128_kc1000_cluster");

    std::iota(book.begin(), book.end(), (uint32_t)0);
    index.load_from_book(book, db_path + "/sift1m_pq128_kc1000_cluster");


    index.load_pq_codebook(db_path + "/sift1m_pq128_kc1000_cluster");
    index.load_cq_codebook(db_path + "/sift1m_pq128_kc1000_cluster");

    query.resize(nq * D);
    auto nested_queries = nest_2d(query, {nq, D});
    std::vector<std::vector<uint32_t>> topw;
    index.top_w_id(50, nested_queries, topw);
    for (size_t n = 0; n < nq; ++n) {
        std::cerr << "Query " << n << " :\n";
        for (const auto& id : topw[n]) {
            std::cerr << id << ", ";
        }
        std::cerr << std::endl;
    }

    // index.set_cluster_vector_path(db_path + "/sift1m_pq128_kc1000_cluster");
    // index.set_cluster_id_path(db_path + "/sift1m_pq128_kc1000_cluster");
    index.finalize();

    return 0;
}