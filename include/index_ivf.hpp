#ifndef INDEX_IVF_H
#define INDEX_IVF_H

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <fstream>
#include "util.hpp"
#include "quantizer.hpp"
#include "distance.hpp"
#include <omp.h>


namespace toy {

/**
 * Configuration structure
 * @param N_ the number of data
 * @param D_ the number of dimensions
 * @param L_ the expected number of candidates involed when searching is performed
 * @param kc the number of coarse quantizer's (nlist) centers. Default: 100
 * @param mc the number of subspace for coarse quantizer. mc must be 1
 * @param dc the dimensions of subspace for coarse quantize and product quantizer. dc must be D_.  dp = D_ / mp
 * @param db_path path to the DB files
 * @param db_prefix the prefix of DB files
 */
struct IVFConfig {
	size_t N_, D_, L_, kc, mc, dc;
    std::string index_path;
    std::string db_path;
    
    explicit IVFConfig(
        size_t N, size_t D, 
        size_t L, 
        size_t kc, 
        size_t mc, 
        size_t dc, 
        std::string index_path, std::string db_path
    );
};

template <typename T> class IndexIVF {
public:
    IndexIVF(const IVFConfig& cfg, size_t nq, bool verbose);

    void Populate(const std::vector<T>& rawdata);
    void LoadCqCodebook(std::string cq_codebook_path);
    void Train(const std::vector<T>& rawdata, int seed, size_t nsamples);
    void LoadIndex(std::string index_path);
    void WriteIndex(std::string index_path);

    // IVF baseline
    void
    QueryBaseline(
        const std::vector<T>& query,
        std::vector<size_t>& nnid,
        std::vector<float>& dist,
        size_t& searched_cnt,
        int topk,
        int L,
        int id, 
        int W
    );
    
private:
    void InsertIvf(const std::vector<T>& rawdata);

    const T* GetSingleCode(size_t list_no, size_t offset) const;
    // Given a long (N * M) codes, pick up n-th code
    // template<typename T>
    const std::vector<T> NthRawVector(const std::vector<T>& long_code, size_t n) const;
    // Member variables
    size_t N_, D_, L_, nq, kc, mc, dc;
    bool verbose_, write_trainset_, is_trained_;

    std::unique_ptr<Quantizer::Quantizer<T>> cq_;

    std::vector<std::vector<float>> centers_cq_;
    std::vector<int> labels_cq_;


    std::vector<std::vector<T>> db_codes_; // binary codes, size nlist
    std::vector<std::vector<int>> posting_lists_;  // (NumList, any)
};


} // namespace toy


#endif