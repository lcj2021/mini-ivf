#include "QueryNode.hpp"

#include <fstream>
#include <unistd.h>

#include "MiniIvfPrint.hpp"

#include <RCF/FileSystem.hpp>

using namespace MiniIVF;


QueryNode::QueryNode(
    const std::string & id,
    const std::pair<std::string, int> & addr
): 
id(id),
num_threads(1), 
querynode_server(RCF::TcpEndpoint(addr.first, addr.second))
{
    querynode_server.bind<I_MiniIvfService>(*this);
    querynode_server.start();

    RCF::setDefaultMaxMessageLength(MiniIVF::MAX_MESSAGE_LENGTH);

    Printer::print("QueryNode start on "+addr.first+":"+std::to_string(addr.second)+"\n", COLOR::GREEN);
}

void QueryNode::setNumThreads(int nt) {
    if (nt < 1) {
        Printer::err("Invalid thread number.\n");
        exit(1);
    }
    num_threads = nt;
    Printer::print("Set "+std::to_string(nt)+" threads on "+id+".\n");
}

/// @brief RPC FUNC: init index
void QueryNode::indexInit(
    size_t N, size_t D, 
    size_t L, 
    size_t kc, size_t kp, 
    size_t mc, size_t mp, 
    size_t dc, size_t dp,
    std::string index_path, std::string db_path
) 
{
    config_ptr = std::make_unique<Toy::IVFPQConfig>(
        N, D, L, kc, kp, mc, mp, dc, dp, index_path, db_path
    );
    index_ptr = std::make_unique<Toy::IndexIVFPQ>(*config_ptr, 2000, false);

    /// @todo Upload Initialize Process.
    std::thread (&QueryNode::uploadInit, this).detach();
}

/// @brief RPC FUNC: segments from file
void QueryNode::loadSegments(
    std::vector<cluster_id_t> book
)
{
    index_ptr->load_from_book(book, config_ptr->db_path);
}

/// @brief RPC FUNC: load code book
void QueryNode::loadCodeBook()
{
    index_ptr->load_index(config_ptr->index_path);
}


/// @brief RPC FUNC: run queries
void QueryNode::runQueries(
    size_t k,
    std::vector<std::vector<vector_dim_t>> queries,
    std::vector<std::vector<cluster_id_t>> topws,
    std::vector<std::vector<vector_id_t>> & topks,
    std::vector<std::vector<asym_dist_t>> & dists
)
{
    Timer timer;
    timer.start();
    index_ptr->top_k_id(k, queries, topws, topks, dists, num_threads);
    timer.stop();
    Printer::print("Time Cost[" + id + "] " + std::to_string(timer.get_time()) + "\n");
}


/// @brief RPC FUNC: add segment file
void QueryNode::addFile(std::string upload_id, std::string file_name)
{
    auto & session = RCF::getCurrentRcfSession();
    auto old_file_name = session.getUploadPath(upload_id);
    auto new_file_name = old_file_name.parent_path() / file_name;
    
    try
    {
        RCF::FsWrappers::rename(old_file_name, new_file_name);
    }
    catch(const std::exception& e)
    {
        Printer::err("[QueryNode::addSegmentFile] ");
        Printer::err(e.what());
        Printer::err("\n");
        exit(1);
    }
    
}


/// @brief Thread Task to close the door and set the upload directory. 
void QueryNode::uploadInit()
{
    sleep(OND_SECOND);
    querynode_server.stop();
    querynode_server.setUploadDirectory(config_ptr->db_path);
    querynode_server.setUploadProgressCallback(
        [](RCF::RcfSession & session, RCF::FileUploadInfo & fuli) {
            Printer::print("Recieving segment from GlobalNode [" + session.getClientAddress().string() + "]\n");
        }
    );
    querynode_server.start();
}
