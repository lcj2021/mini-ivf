#include "GlobalNode.hpp"
#include <binary_io.hpp>
#include "MiniIvfIni.hpp"

size_t D, nb, mp, ncentroids;
size_t nq, nprobe, k, batch_size, num_threads, num_batchs_every_req, global_caches;

std::string index_path;
std::string db_path;
std::string query_path;

std::vector<vector_dim_t> query;
std::vector<cluster_id_t> gt;

static void recall(const std::vector<std::vector<std::pair<MiniIVF::vector_id_t, MiniIVF::asym_dist_t>>>& res);
static std::vector<std::pair<std::string, int>> get_querynode_info(MiniIVF::inifile & ini);



int main(int argc, char** argv)
{
    /// @brief READ CONFIG FILE
    assert(argc == 2);

    MiniIVF::inifile ini(argv[1]);
    auto section_data = ini.section("data");
    auto section_task = ini.section("task");
    auto section_querynode = ini.section("querynode");

    /// @todo load data info
    D = section_data.get_int("D");
    nb = section_data.get_int("nb");
    mp = section_data.get_int("mp");
    ncentroids = section_data.get_int("ncentroids");
    index_path = section_data.get_string("index_path");
    db_path = section_data.get_string("db_path");
    query_path = section_data.get_string("query_path");

    /// @todo load task info
    nq = section_task.get_int("nq");
    nprobe = section_task.get_int("nprobe");
    k = section_task.get_int("k");
    batch_size = section_task.get_int("batch_size");
    num_threads = section_task.get_int("num_threads");
    global_caches = section_task.get_int("global_caches");

    MiniIVF::GlobalNode gn;
    gn.setQueryNode( get_querynode_info(ini) );
    gn.indexInit( 
        Toy::IVFPQConfig ( 
            nb, D, nb, ncentroids, 256, 1, mp, D, D / mp,
            index_path, 
            db_path
        ) 
    );
    gn.loadPostingListsSize();
    gn.setNumThreads(num_threads);
    
    load_from_file_binary(query, query_path + "/query.fvecs");
    load_from_file_binary(gt, query_path + "/query_groundtruth.ivecs");


    size_t num_batches = (nq + batch_size - 1) / batch_size;
    Timer timer_query;

    /*

    {
        gn.loadBalance(MiniIVF::BalanceMode::Normal);

        std::vector<std::vector<std::pair<MiniIVF::vector_id_t, MiniIVF::asym_dist_t>>> res;
        timer_query.start();

        for (size_t i = 0; i < num_batches; i++)
        {
            auto beg = query.begin() + i * batch_size * D;
            auto end = beg + std::min(batch_size, nq - i * batch_size) * D;
            auto batch_queries = std::vector<vector_dim_t>(beg, end);
            auto nested_queries = nest_2d(batch_queries, {batch_size, D});
            auto res = gn.runQueries(k, nprobe, nested_queries);
        }

        timer_query.stop();    

        std::cout << "Normal: " << timer_query.get_time() << " seconds.\n";
        timer_query.reset();
        // recall(res);
    }

    std::cout << "-----------------------------------------------------------" << std::endl;

    {
        gn.setGlobalCaches(global_caches);
        gn.loadBalance(MiniIVF::BalanceMode::BestFitHybrid);
        
        timer_query.start();

        for (size_t i = 0; i < num_batches; i++)
        {
            auto beg = query.begin() + i * batch_size * D;
            auto end = beg + std::min(batch_size, nq - i * batch_size) * D;
            auto batch_queries = std::vector<vector_dim_t>(beg, end);
            auto nested_queries = nest_2d(batch_queries, {batch_size, D});
            auto res = gn.runQueries(k, nprobe, nested_queries);
        }
        
        timer_query.stop();    

        std::cout << "BestFitHybrid: " << timer_query.get_time() << " seconds.\n";
        timer_query.reset();
        // recall(res);
    }

    */

    gn.uploadSegmentFile({"slave0", 50000}, 0);

    return  0;
}


void recall(const std::vector<std::vector<std::pair<MiniIVF::vector_id_t, MiniIVF::asym_dist_t>>>& res)
{
    int n_ok = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<int> S(gt.begin() + q * 100, gt.begin() + q * 100 + k);
        for (int i = 0; i < k; ++i)
            if (S.count(res[q][i].first))
                n_ok++;
    }
    std::cout << "Recall@" << k << ": " << (double)n_ok / (nq * k) << '\n';
    printf("PQ%lu, kc%d, W%d\n", (unsigned long)mp, (int)ncentroids, (int)nprobe);
}


std::vector<std::pair<std::string, int>> get_querynode_info(MiniIVF::inifile & ini)
{
    std::vector<std::pair<std::string, int>> querynodes;
    auto & section_querynode = ini.section("querynode");
    size_t num_querynodes = section_querynode.get_int("num_querynodes");
    
    for (size_t i = 0; i < num_querynodes; i++) {
        std::string host = section_querynode.get_string("a"+std::to_string(i));
        int port = section_querynode.get_int("p"+std::to_string(i));
        querynodes.emplace_back(std::make_pair(host, port));
    }

    return querynodes;
}
