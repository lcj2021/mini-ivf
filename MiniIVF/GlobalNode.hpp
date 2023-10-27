#ifndef INCLUDE_GLOBALNODE_HPP
#define INCLUDE_GLOBALNODE_HPP

#include <vector>
#include <utility>
#include <string>
#include <unordered_map>

#include "MiniIvfConst.hpp"
#include "MiniIvfInterface.hpp"

#include <RCF/RCF.hpp>

#include <index_ivfpq.hpp>


namespace MiniIVF {

enum BalanceMode {
    Normal, 
    BestFitSize,
    BestFitPop,
    BestFitHybrid
};

class GlobalNode {
private: /// variables
    // index
    std::unique_ptr<Toy::IVFPQConfig> config_ptr;
    std::unique_ptr<Toy::IndexIVFPQ> index_ptr;
    std::vector<std::pair<std::string, int>> querynodes;
    std::vector<querynode_id_t> querybook;

    // for balance usage
    std::vector<history_score_t> popularity;
    std::vector<size_t> posting_lists_size;
    int num_threads;

    // Caching
    size_t global_caches;


public: /// methods

    explicit GlobalNode();

    void setNumThreads(int nt);

    void setGlobalCaches(size_t caches);

    /// @brief set query node by calling this method.
    void setQueryNode(
        const std::vector<std::pair<std::string, int>> & qa);

    /// @brief init global node index and query node indexes.
    /// @param cfg index config
    void indexInit(Toy::IVFPQConfig cfg);

    /// @brief balance segments to each querynode.
    /// @param bm balance mode
    void loadBalance(BalanceMode bm);


    /// @brief clear all popularity infomation.
    void clearHistory();

    void loadPostingListsSize();

    /// @brief run queries
    std::vector<std::vector<std::pair<vector_id_t, asym_dist_t>>>
    runQueries(
        size_t k,
        size_t w,
        const std::vector<std::vector<vector_dim_t>> & queries
    );


    /// @brief transfer file to QueryNode.
    void uploadSegmentFile(
        const std::pair<std::string, int> & addr, 
        cluster_id_t cid
    );


private: /// methods

    /// @brief private RPC Funtion in local expression: loadSegments
    void loadSegments(
        const std::pair<std::string, int> & addr,
        const std::vector<cluster_id_t> & book
    );

    /// @brief private RPC Funtion in local expression: loadCodeBook    
    void loadCodeBook(const std::pair<std::string, int> & addr);

    /// @brief private RPC Funtion in local expression: indexInit
    void indexInit(
        const std::pair<std::string, int> & addr,
        const Toy::IVFPQConfig & cfg
    );

    /// @brief private RPC Function in local expression: runQueries
    void
    runQueries(
        const std::pair<std::string, int> & addr,
        size_t k,
        const std::vector<std::vector<vector_dim_t>> & queries,
        const std::vector<std::vector<cluster_id_t>> & topw,
        std::vector<std::vector<vector_id_t>> & topks,
        std::vector<std::vector<asym_dist_t>> & dists
    );

};

};

#endif // ! INCLUDE_GLOBALNODE_HPP