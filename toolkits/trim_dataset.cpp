#include <random>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "binary_io.hpp"
#include "index_ivfpq.hpp"
#include "quantizer.hpp"
#include "util.hpp"

using string = std::string;

string suffix = ".bvecs";

int main(int argc, char* argv[]) {
    assert(argc == 2);
    size_t n_vector_out = atoi(argv[1]);

    string prefix_in = "sift1b/";
    string prefix_out = "sift" + to_string_with_units(n_vector_out) + "/";
    string in_db_path = "/dk/anns/dataset/" + prefix_in;
    string out_db_path = "/dk/anns/dataset/" + prefix_out;

    modify_path(in_db_path);
    modify_path(out_db_path);
    string mkdir_cmd = "mkdir -p " + out_db_path;
    if (system(mkdir_cmd.data())) {
        std::cerr << "mkdir " << out_db_path << " failed!\n";
        throw;
    }

    /// @warning DO NOT OVERWRITE original database!!!
    if (in_db_path == out_db_path) {
        std::cerr << "DO NOT OVERWRITE original database!!!\n";
        throw;
    }

    std::cerr << "out_db_path: " << out_db_path << '\n'
             << "in_db_path: " << in_db_path << '\n';
    std::vector<uint8_t> database;
    auto [n, d] = load_from_file_binary<uint8_t>(database, in_db_path + "base" + suffix, n_vector_out);

    assert(n == n_vector_out);
    write_to_file_binary(database, {n, d}, out_db_path + "base" + suffix);

    return 0;
}