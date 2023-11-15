#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <numeric>
#include <distance.hpp>
#include <cassert>
#include <memory>
#include <unordered_set>
#include <algorithm>

#include <ivf/index_ivf.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/vector_io.hpp>

const std::string name              = "IVF4SIFT1M";
size_t nt                           = 1'000'000;   
size_t kc                           = 4096;
size_t kp                           = 256;
size_t mp                           = 64;
size_t k                            = 10;

const std::string dataset_path      = "../../data/dataset/sift1m/";
const std::string index_path        = "../../data/index/sift1m/ivf/";
const std::string db_path           = "../../data/db/sift1m/ivf/";
const std::string queries_path      = "/dk/anns/query/sift1m/";

using vector_dimension_t = float;

int main(int argc, char *argv[]) 
{

    assert(argc == 2 || argc == 3);

    std::vector<vector_dimension_t> database;
    auto [nb, D] = utils::VectorIO<vector_dimension_t>::LoadFromFile(database, dataset_path + "base.fvecs");
    std::cout << "database size: " << nb << " x " << D << std::endl;
    std::vector<vector_dimension_t> queries;
    auto [nq, D1] = utils::VectorIO<vector_dimension_t>::LoadFromFile(queries, queries_path + "query.fvecs");
    std::cout << "queries size: " << nq << " x " << D1 << std::endl;
    assert( D == D1 );
    auto queries_nest = utils::Resize<vector_dimension_t>::Nest(queries, nq, D1);
    std::vector<index::vector_id_t> gt;
    auto [n_gt, d_gt] = utils::VectorIO<index::vector_id_t>::LoadFromFile(gt, queries_path + "gt.ivecs");
    std::cout << "ground truth size: " << n_gt << " x " << d_gt << std::endl;

    size_t nprobe = atoi(argv[1]);

    auto ivf = new index::ivf::IndexIVF<float> (
        nb, D, 6700, kc, 
        index_path, db_path, name,
        index::IndexStatus::LOCAL
    );

    std::cout << "Test from " << ivf->GetName() << std::endl;

    std::vector<std::vector<index::vector_id_t>>    nnid(nq, std::vector<index::vector_id_t>(k));
    std::vector<std::vector<float>>                 dists(nq, std::vector<float>(k));

    utils::STimer timer;

    if (argc == 3 && std::string(argv[2]) == "check") {
        puts("Start to check...");
        ivf->LoadIndex();
        ivf->LoadSegments();

        // Recall@k
        ivf->SetNumThreads(48);
        timer.Start();
        ivf->Search(k, nprobe, queries_nest, nnid, dists);
        timer.Stop();
        std::cout << "Search time: " << timer.GetTime() << " seconds" << std::endl;

        size_t n_ok = 0;
        for (size_t q = 0; q < nq; q++)
        {
            std::unordered_set<index::vector_id_t> S(gt.begin() + q * d_gt, gt.begin() + q * d_gt + k);
            for (const auto & id: nnid[q])
            {
                if (S.count(id)) 
                    n_ok ++;
            }
        }

        std::cout << "Recall@" << k << ": " << (double)n_ok / (nq * k) << std::endl;
    
    } else {

        puts("Index init start...");

        ivf->SetTrainingConfig(nt, 123);
        ivf->Train(database);
        ivf->Populate(database);

        std::cout << "Index init done." << std::endl;

        // Write index and PQCODE
        ivf->WriteIndex();
        ivf->WritePostingLists();
        ivf->WriteSegments();

        std::cout << "Index write done." << std::endl;

    }

    return 0;
}