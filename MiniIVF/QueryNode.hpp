#ifndef INCLUDE_QUERYNODE_HPP
#define INCLUDE_QUERYNODE_HPP

#include <unordered_map>
#include <thread>
#include <string>
#include <iostream>
#include <utility>

#include <RCF/RCF.hpp>

#include "MiniIvfType.hpp"
#include "MiniIvfConst.hpp"
#include "MiniIvfInterface.hpp"

#include <binary_io.hpp>
#include <quantizer.hpp>
#include <util.hpp>
#include <index_ivfpq.hpp>

namespace MiniIVF {


class QueryNode {
private: // variables
    std::unique_ptr<Toy::IVFPQConfig> config_ptr;
    RCF::RcfServer querynode_server;
    std::unique_ptr<Toy::IndexIVFPQ> index_ptr;
    int num_threads;
    std::string id;

public: // methods
    explicit QueryNode(
        const std::string & id,
        const std::pair<std::string, int> & addr
    );

    void setNumThreads(int nt);

    void indexInit(
        size_t N, size_t D, 
        size_t L, 
        size_t kc, size_t kp, 
        size_t mc, size_t mp, 
        size_t dc, size_t dp,
        std::string index_path, std::string db_path
    );

    void loadSegments(
        std::vector<cluster_id_t> book
    );

    void loadCodeBook();

    void runQueries(
        size_t k,
        std::vector<std::vector<vector_dim_t>> queries,
        std::vector<std::vector<cluster_id_t>> topws,
        std::vector<std::vector<vector_id_t>> & topks,
        std::vector<std::vector<asym_dist_t>> & dists
    );

    void addFile(std::string upload_id, std::string file_name);


private: // methods
    void uploadInit();

};

};

#endif // ! INCLUDE_QUERYNODE_HPP