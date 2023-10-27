#include "GlobalNode.hpp"
#include "MiniIvfPrint.hpp"
#include "MiniIvfType.hpp"
#include "MiniIvfConst.hpp"
#include <RCF/ClientTransport.hpp>
#include <RCF/FileUpload.hpp>
#include <RCF/Uuid.hpp>
#include <thread>
#include <mutex>
#include <util.hpp>


using namespace MiniIVF;


GlobalNode::GlobalNode(): 
num_threads(1), 
querybook(MAX_CLUSTER_NUM, NULL_QUERYNODE), 
popularity(MAX_CLUSTER_NUM, 0),
global_caches(0)
{
    RCF::setDefaultMaxMessageLength(MiniIVF::MAX_MESSAGE_LENGTH);
}



void GlobalNode::setNumThreads( int nt )
{
    if (nt < 1) {
        Printer::err("Invalid thread number.\n");
        exit(1);
    }
    num_threads = nt;
    Printer::print("Set "+std::to_string(num_threads)+" threads.\n");
}



void GlobalNode::setQueryNode(
    const std::vector<std::pair<std::string, int>> & qa
)
{
    querynodes = qa;
    Printer::print("Set "+std::to_string(querynodes.size())+" query nodes.\n");
}


void GlobalNode::setGlobalCaches(size_t caches)
{
    global_caches = caches;
    Printer::print("Set global cache size: "+std::to_string(global_caches)+"\n");
}



void GlobalNode::indexInit( Toy::IVFPQConfig cfg )
{
    if (cfg.index_path.back() != '/') {
        cfg.index_path += "/";
    }
    if (cfg.db_path.back() != '/') {
        cfg.db_path += "/";
    }
    config_ptr = std::make_unique<Toy::IVFPQConfig>(cfg);
    index_ptr = std::make_unique<Toy::IndexIVFPQ>(cfg, 2000, false);
    
    size_t querynodes_size = querynodes.size();
    auto init_task = [&](size_t idx) {
        if (idx == querynodes_size) {
            index_ptr->load_index(cfg.index_path);
        }
        else {
            const auto & addr = querynodes[idx];
            indexInit(addr, cfg);
            loadCodeBook(addr);
        }
    };

    { /// Thread Pool Start
        std::vector<std::thread> pool;
        for (size_t i = 0; i <= querynodes_size; i++) {
            pool.emplace_back(std::thread{init_task, i});
        }
        for (auto & t: pool) t.join();
    }
}



void GlobalNode::clearHistory()
{
    popularity.assign(MAX_CLUSTER_NUM, 0);
    Printer::print("Clear popularity successfully.\n");
}



void GlobalNode::loadPostingListsSize()
{
    std::string ul_suffix = ".ulvecs";
    load_from_file_binary(posting_lists_size, config_ptr->db_path + "posting_lists_lens" + ul_suffix);
    if (posting_lists_size.size() != config_ptr->kc)
    {
        Printer::err("Invalid posting lists size.\n");
        exit(1);
    }
    Printer::print("Load posting lists size successfully.\n");
}


void GlobalNode::loadBalance( BalanceMode bm )
{
    Printer::print("Load balance start.\n");

    if (querynodes.empty()) {
        Printer::err("You should set querynodes firstly.\n");
        exit(1);
    }
    /// @todo clear all qureybook records.
    querybook.assign(MAX_CLUSTER_NUM, NULL_QUERYNODE);

    std::vector<std::vector<cluster_id_t>> books(querynodes.size());
    size_t num_segments = config_ptr->kc;
    size_t querynodes_size = querynodes.size();

    /// @todo helper for bestfit algorithm
    std::vector<history_score_t> scores(querynodes_size, 0);
    auto get_min_querynode = [&]() {
        history_score_t min_score = std::numeric_limits<history_score_t>::max();
        size_t min_idx = 0;
        for (size_t i = 0; i < querynodes_size; i++) {
            if (scores[i] < min_score) {
                min_score = scores[i];
                min_idx = i;
            }
        }
        return min_idx;
    };

    std::vector<cluster_id_t> sorted_cluster_ids(num_segments);
    for (size_t i = 0; i < num_segments; i++) {
        sorted_cluster_ids[i] = i;
    }

    switch (bm)
    {
        /// @brief Normal Mode: divide segments into each querynode, while the numbers of each querynode are average.
        case BalanceMode::Normal:
        { 
            for (size_t i = 0, j = 0; i < num_segments; i++, j = (j + 1) % querynodes_size) {
                querybook[i] = j; // i-th segment been packed into j-th querynode.
                books[j].emplace_back(i);
                popularity[i] ++; // i-th segment hit.
            }
        } 
        break;

        /// @brief BestFitSize Mode: divide segments into QNs, while using 'BestFit' algorithm by posting-list-size.
        case BalanceMode::BestFitSize:
        {
            std::sort(sorted_cluster_ids.begin(), sorted_cluster_ids.end(), 
                [&](cluster_id_t a, cluster_id_t b)
                {
                    return posting_lists_size[a] > posting_lists_size[b];
                }
            );
            for (size_t i = 0; i < num_segments; i++)
            {
                cluster_id_t id = sorted_cluster_ids[i];
                size_t min_idx = get_min_querynode();
                querybook[id] = min_idx;
                scores[min_idx] += posting_lists_size[id];
                books[min_idx].emplace_back(id);
            }
        }
        break;

        /// @brief BestFitSize Mode: divide segments into QNs, while using 'BestFit' algorithm by popularity.
        case BalanceMode::BestFitPop:
        {
            std::sort(sorted_cluster_ids.begin(), sorted_cluster_ids.end(), 
                [&](cluster_id_t a, cluster_id_t b)
                {
                    return popularity[a] > popularity[b];
                }
            );
            for (size_t i = 0; i < num_segments; i++)
            {
                cluster_id_t id = sorted_cluster_ids[i];
                size_t min_idx = get_min_querynode();
                querybook[id] = min_idx;
                scores[min_idx] += popularity[id];
                books[min_idx].emplace_back(id);
            }
        }
        break;

        /// @brief BestFitHybrid Mode: divide segments into QNs, while using 'BestFit' algorithm by 'pop x posting-lists-size'.
        case BalanceMode::BestFitHybrid:
        {
            std::vector<history_score_t> product(num_segments);
            for (size_t i = 0; i < num_segments; i++)
            {
                product[i] = popularity[i] * posting_lists_size[i];
            }
            std::sort(sorted_cluster_ids.begin(), sorted_cluster_ids.end(), 
                [&](cluster_id_t a, cluster_id_t b)
                {
                    return product[a] > product[b];
                }
            );
            for (size_t i = 0; i < num_segments; i++)
            {
                cluster_id_t id = sorted_cluster_ids[i];
                size_t min_idx = get_min_querynode();
                querybook[id] = min_idx;
                scores[min_idx] += product[id];
                books[min_idx].emplace_back(id);
            }
        }
        break;

        default:
            Printer::err("Invalid balance mode.\n");
            exit(1);
    }

    /// @todo glboal caching
    std::vector<cluster_id_t> book_global;
    std::partial_sort(sorted_cluster_ids.begin(), sorted_cluster_ids.begin()+global_caches, sorted_cluster_ids.end(), 
        [&](cluster_id_t a, cluster_id_t b) { return popularity[a] > popularity[b]; }
    );
    for (size_t i = 0; i < global_caches; i++)
    {
        cluster_id_t cid = sorted_cluster_ids[i];
        querybook[cid] = GLOBAL_QUERYNODE;
        book_global.emplace_back(cid);
    }

    auto load_segments_task = [&](size_t idx) {
        if (idx == querynodes_size) {
            index_ptr->load_from_book(book_global, config_ptr->db_path);
            Printer::print(std::to_string(book_global.size()) + " segments have been loaded into global cache.\n");
        }
        else {
            loadSegments(querynodes[idx], books[idx]);
            char msg[128];
            sprintf(msg, "%d segments have been loaded into %s:%d.\n", 
                    (int)books[idx].size(), querynodes[idx].first.c_str(), querynodes[idx].second);
            Printer::print(msg);
        }
    };

    { /// Thread Pool Start!
        std::vector<std::thread> pool;
        for (size_t i = 0; i <= querynodes_size; i++)
        {
            pool.emplace_back(load_segments_task, i);
        }
        for (auto & t: pool) t.join();
    } /// Thread Pool End!

    Printer::print("Load balance end.\n");
}



std::vector<std::vector<std::pair<vector_id_t, asym_dist_t>>>
GlobalNode::runQueries(
    size_t k,
    size_t w,
    const std::vector<std::vector<vector_dim_t>> & queries
)
{
    Printer::print("Searching started.\n");
    size_t querynodes_size = querynodes.size();

    std::vector<std::vector<cluster_id_t>> topw;
    std::vector<std::vector<std::vector<cluster_id_t>>> topws (
        querynodes.size(),
        std::vector<std::vector<cluster_id_t>> (queries.size())
    );
    std::vector<std::vector<cluster_id_t>> topws_global(queries.size());
    
    /// @brief to get topw from index
    index_ptr->top_w_id(w, queries, topw, num_threads);
    for (size_t i = 0; i < queries.size(); i++) {
        const auto & topw4query = topw[i];
        for (auto cid: topw4query) {
            auto qid = querybook[cid];
            if (qid == GLOBAL_QUERYNODE)
            {
                topws_global[i].emplace_back(cid);
            }
            else {
                topws[qid][i].emplace_back(cid);
            }
            popularity[cid]++;
        }
    }

    /// @todo Result collecting.
    std::vector<std::vector<std::pair<vector_id_t, asym_dist_t>>> scores(queries.size());
    std::mutex mtx;

    auto run_queries_task = [&](size_t idx) {
        std::vector<std::vector<vector_id_t>> topks(queries.size());
        std::vector<std::vector<asym_dist_t>> dists(queries.size());
        if (idx == querynodes_size) {
            Timer timer;
            timer.start();
            index_ptr->top_k_id(k, queries, topws_global, topks, dists, num_threads);
            timer.stop();
            Printer::print("Global Time Cost: " + std::to_string(timer.get_time()) + "\n");
        }
        else {
            runQueries(querynodes[idx], k, queries, topws[idx], topks, dists);
        }
        /// merge result in pair format.
        for (size_t j = 0; j < queries.size(); j++) {
            for (size_t k = 0; k < topks[j].size(); k++) {
                auto elem = std::make_pair(topks[j][k], dists[j][k]);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    scores[j].emplace_back(elem);
                }
            }
        }
    };

    { /// Thread Pool Start
        std::vector<std::thread> pool;
        for (size_t i = 0; i <= querynodes_size; i++)
        {
            pool.emplace_back(std::thread(run_queries_task, i));
        }
        for (auto & t: pool) t.join();
    } /// Thread Pool End

    /// @todo merge all querynodes' results.
    auto topk_sort_task = [&](size_t idx) {
        auto & score = scores[idx];
        int actual_k = std::min(k, score.size());
        std::partial_sort(score.begin(), score.begin()+actual_k, score.end(),
            [](const std::pair<vector_id_t, asym_dist_t> & a, const std::pair<vector_id_t, asym_dist_t> & b) {
                return a.second < b.second;
            }
        );
    };

    { /// Thread Pool Start
        std::vector<std::thread> pool;
        for (size_t i = 0; i < queries.size(); i++)
        {
            pool.emplace_back(std::thread(topk_sort_task, i));
        }
        for (auto & t: pool) t.join();
    } /// Thread Pool End

    Printer::print("Searching finished.\n");

    return scores;
}



/// @brief private RPC Funtion in local expression: loadSegments
void GlobalNode::loadSegments(
    const std::pair<std::string, int> & addr,
    const std::vector<cluster_id_t> & book
) 
{
    RCF::RcfInit rcf_init;
    try
    {
        RcfClient<I_MiniIvfService>(RCF::TcpEndpoint(addr.first, addr.second)
        ).loadSegments(book);
    }
    catch(const std::exception& e)
    {
        Printer::err("[GlobalNode::LoadSegments] ");
        Printer::err(e.what());
        Printer::err("\n");
        exit(1);
    } 
}


/// @brief private RPC Funtion in local expression: loadCodeBook
void GlobalNode::loadCodeBook(
    const std::pair<std::string, int> & addr
)
{
    RCF::RcfInit rcf_init;
    try
    {
        RcfClient<I_MiniIvfService>(RCF::TcpEndpoint(addr.first, addr.second)
        ).loadCodeBook();
    }
    catch(const std::exception& e)
    {
        Printer::err("[GlobalNode::LoadCodeBook] ");
        Printer::err(e.what());
        Printer::err("\n");
        exit(1);
    }
}


/// @brief private RPC Funtion in local expression: indexInit
void GlobalNode::indexInit(
    const std::pair<std::string, int> & addr,
    const Toy::IVFPQConfig & cfg
)
{
    RCF::RcfInit rcf_init;
    try
    {
        RcfClient<I_MiniIvfService>(RCF::TcpEndpoint(addr.first, addr.second)
        ).indexInit(
            cfg.N_, cfg.D_, cfg.L_, cfg.kc, cfg.kp, 
            cfg.mc, cfg.mp, cfg.dc, cfg.dp,
            cfg.index_path, cfg.db_path
        );
    }
    catch(const std::exception& e)
    {
        Printer::err("[GlobalNode::indexInit] ");
        Printer::err(e.what());
        Printer::err("\n");
        exit(1);
    }
    /// @todo Sleep 2 seconds to wait QNs to set their upload directories.
    sleep((MiniIVF::OND_SECOND)<<1);
}



/// @brief private RPC Function in local expression: runQueries
void
GlobalNode::runQueries(
    const std::pair<std::string, int> & addr,
    size_t k,
    const std::vector<std::vector<vector_dim_t>> & queries,
    const std::vector<std::vector<cluster_id_t>> & topws,
    std::vector<std::vector<vector_id_t>> & topks,
    std::vector<std::vector<asym_dist_t>> & dists
)
{
    RCF::RcfInit rcf_init;
    try
    {
        RcfClient<I_MiniIvfService> client (RCF::TcpEndpoint(addr.first, addr.second));
        auto & client_stub = client.getClientStub();
        client_stub.setRemoteCallTimeoutMs(MiniIVF::MAX_REMOTE_CALL_TIMEOUT_MS);    
        client.runQueries(k, queries, topws, topks, dists);
    }
    catch(const std::exception& e)
    {
        Printer::err("[GlobalNode::runQueries] ");
        Printer::err(e.what());
        Printer::err("\n");
        exit(1);
    }
}


void GlobalNode::uploadSegmentFile(
    const std::pair<std::string, int> & addr, 
    cluster_id_t cid
)
{
    if (config_ptr == nullptr)
    {
        Printer::err("[GlobalNode::uploadSegmentFile] Configure is uninitialized\n");
        exit(1);
    }

    if (cid < 0 || cid >= config_ptr->kc)
    {
        Printer::err("[GlobalNode::uploadSegmentFile] cid out of range\n");
        exit(1);
    }

    const std::string segment_file_prefix = "pqcode_";
    const std::string segment_file_suffix = ".ui8vecs";

    std::string segment_file_name = segment_file_prefix + std::to_string(cid) + segment_file_suffix;
    /// Assert that db path has suffix '/'.
    RCF::Path segment_file_path = config_ptr->db_path + segment_file_name;

    RCF::RcfInit rcf_init;
    try
    {
        RcfClient<I_MiniIvfService> client (RCF::TcpEndpoint(addr.first, addr.second));
        std::string upload_id = RCF::generateUuid();

        RCF::FileTransferOptions options;
        options.mProgressCallback = [&addr] ( const RCF::FileTransferProgress & progress, RCF::RemoteCallAction & action ) {
            double percent = (double)progress.mBytesTransferredSoFar / (double)progress.mBytesTotalToTransfer;
            Printer::print("Upload segment to [" + addr.first + ":" + std::to_string(addr.second) + "] " + std::to_string(percent * 100) + "%\n");
        };

        client.getClientStub().uploadFile(upload_id, segment_file_path, &options);
        client.addFile(upload_id, segment_file_name);
    }
    catch(const std::exception& e)
    {
        Printer::err("[GlobalNode::uploadSegmentFile] ");
        Printer::err(e.what());
        Printer::err("\n");
        exit(1);
    }
}
