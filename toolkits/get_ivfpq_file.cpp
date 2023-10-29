#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.hpp"
#include "index_ivfpq.hpp"
#include "quantizer.hpp"
#include "util.hpp"

using string = std::string;

size_t D;           // dimension of the vectors to index
size_t nb;          // size of the database we plan to index
size_t nt = 100'000;          // make a set of nt training vectors in the unit cube (could be the database)
size_t mp = 128;
size_t nq = 10'000;
int ncentroids = 2000;
int nprobe = ncentroids;

string prefix = "sift1m/";
string suffix = string("nt") + ToStringWithUnits(nt)
    + string("_pq") + std::to_string(mp) 
    + string("_kc") + std::to_string(ncentroids);

string out_db_path = "/RF/dataset/" + prefix + suffix;
string index_path = "/RF/index/" + prefix + suffix;
string db_path = "/RF/dataset/" + prefix;

int main() {
    ModifyPath(index_path);
    ModifyPath(db_path);
    ModifyPath(out_db_path);

    std::cerr << out_db_path << '\n' << index_path << '\n' << db_path << '\n';
    std::vector<float> database;
    std::tie(nb, D) = LoadFromFileBinary<float>(database, db_path + "base.fvecs");

    std::vector<float> query;
    LoadFromFileBinary<float>(query, db_path + "query.fvecs");

    std::vector<int> gt;
    LoadFromFileBinary<int>(gt, db_path + "query_groundtruth.ivecs");

    toy::IVFPQConfig cfg(
        nb, D, nb, 
        ncentroids, 256, 
        1, mp, 
        D, D / mp, 
        index_path, out_db_path
    );
    toy::IndexIVFPQ index(cfg, nq, true);

    index.Train(database, 123, nt);
    index.WriteIndex(index_path);
    index.LoadIndex(index_path);
    index.Populate(database);

    index.SetClusterVectorPath(out_db_path);
    index.SetClusterIdPath(out_db_path);
    index.Finalize();

    return 0;
}