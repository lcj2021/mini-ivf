#ifndef INCLUDE_INDEX_IVFPQ_HPP
#define INCLUDE_INDEX_IVFPQ_HPP

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <fstream>

#include "util.hpp"
#include "quantizer.hpp"
#include "kmeans.hpp"
#include "binary_io.hpp"
#include "distance.hpp"

#include <omp.h>


namespace toy {

/**
 * Configuration structure
 * @param N_ the number of data
 * @param D_ the number of dimensions
 * @param W_ the number of bucket involed when searching is performed
 * @param L_ the expected number of candidates involed when searching is performed
 * @param kc, kp the number of coarse quantizer (nlist) and product quantizer's centers (1 << nbits). Default: 100, 256
 * @param mc, mp the number of subspace for coarse quantizer and product quantizer. mc must be 1
 * @param dc, dp the dimensions of subspace for coarse quantize and product quantizer. dc must be D_.  dp = D_ / mp
 * @param db_path path to the DB files
 * @param db_prefix the prefix of DB files
 */
class IVFPQConfig {
public:
	size_t N_, D_, L_, kc, kp, mc, mp, dc, dp;
    std::string index_path;
    std::string db_path;

    explicit IVFPQConfig(
        size_t N, size_t D, 
        size_t L, 
        size_t kc, size_t kp, 
        size_t mc, size_t mp, 
        size_t dc, size_t dp,
        std::string index_path, std::string db_path
    );
};

struct DistanceTable {
    // Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
    DistanceTable() {}
    DistanceTable(size_t M, size_t Ks) : kp(Ks), data_(M * Ks) {}
    void set_value(size_t m, size_t ks, float val) {
        data_[m * kp + ks] = val;
    }
    float get_value(size_t m, size_t ks) const {
        return data_[m * kp + ks];
    }
    size_t kp;
    std::vector<float> data_;
};

template <typename T> class IndexIVFPQ {
public:
    IndexIVFPQ(const IVFPQConfig& cfg, size_t nq, bool verbose);

    void Populate(const std::vector<T>& rawdata);
    void LoadFromBook(const std::vector<uint32_t>& book, std::string cluster_path);
    void LoadPqCodebook(std::string pq_codebook_path);
    void LoadCqCodebook(std::string cq_codebook_path);
    void Train(const std::vector<T>& rawdata, int seed, size_t nsamples = 0);
    void LoadIndex(std::string index_path);
    void WriteIndex(std::string index_path);
    void Finalize();

    
    void
    TopWId(
        int w, 
        const std::vector<std::vector<T>>& queries,
        std::vector<std::vector<uint32_t>>& topw, 
        int num_threads
    );

    void 
    TopKId(
        int k, 
        const std::vector<std::vector<T>>& queries, 
        const std::vector<std::vector<uint32_t>>& topw,
        std::vector<std::vector<uint32_t>>& topk_id,
        std::vector<std::vector<float>>& topk_dist,
        int num_threads
    );

    // IVFPQ baseline
    void QueryBaseline(
        const std::vector<T>& query,
        std::vector<size_t>& nnid,
        std::vector<float>& dist,
        size_t& searched_cnt,
        int topk,
        int L,
        int id,
        int W
    );

    // For observation
    void QueryObs(
        const std::vector<T>& query,
        const std::vector<int>& gt,
        std::vector<size_t>& nnid,
        std::vector<float>& dist,
        size_t& searched_cnt,
        int topk,
        int L,
        int id
    );

    void SetClusterVectorPath(std::string cluster_vector_path);

    void SetClusterIdPath(std::string cluster_id_path);

    void SetTrainsetPath(std::string trainset_path, int trainset_type);

    void ShowStatistics();

private:
    void WriteTrainset();
    void WriteClusterVector();
    void WriteClusterId();

    void InsertIvf(const std::vector<T>& rawdata);
    DistanceTable DTable(const std::vector<T>& vec) const;
    float ADist(const DistanceTable& dtable, const std::vector<uint8_t>& code) const;
    float ADist(const DistanceTable& dtable, size_t list_no, size_t offset) const;

    // Given a long (N * M) codes, pick up n-th code
    const T* NthRawVector(const T* long_code_ptr, size_t n) const;

    // Member variables
    size_t N_, D_, L_, nq, kc, kp, mc, mp, dc, dp;
    bool verbose_, is_trained_;

    std::string write_trainset_path_, write_cluster_vector_path_, write_cluster_id_path_;
    int write_trainset_type_;

    std::unique_ptr<Quantizer::Quantizer<T>> cq_, pq_;

    std::vector<std::vector<float>> centers_cq_;
    std::vector<int> labels_cq_;

    std::vector<std::vector<std::vector<float>>> centers_pq_;
    std::vector<std::vector<int>> labels_pq_;


    std::vector<std::vector<uint8_t>> db_codes_; // binary codes, size nlist
    std::vector<std::vector<uint32_t>> posting_lists_;  // (NumList, any)
};


} // namespace toy


#endif