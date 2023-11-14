#ifndef INCLUDE_INDEX_IVFPQ_HPP
#define INCLUDE_INDEX_IVFPQ_HPP

#include <ivf/ivf_base.hpp>
#include <vector>
#include <quantizer.hpp>


namespace index {

namespace ivf {


// Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
class DistanceTable {
public:
    // Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
    explicit DistanceTable(size_t M, size_t Ks);

    inline void SetValue(size_t m, size_t ks, float val);

    inline float GetValue(size_t m, size_t ks) const ;

private:
    size_t kp_;
    std::vector<float> data_;
    
};



/**
 * Configuration structureL_1
 * @param N_ the number of data
 * @param D_ the number of dimensions
 * @param L_ the expected number of candidates involed when searching is performed
 * @param kc, kp the number of coarse quantizer (nlist) and product quantizer's centers (1 << nbits). Default: 100, 256
 * @param mc, mp the number of subspace for coarse quantizer and product quantizer. mc must be 1
 * @param dc, dp the dimensions of subspace for coarse quantize and product quantizer. dc must be D_.  dp = D_ / mp
 * @param index_path path to the index files: pq_centers, cq_centers
 * @param db_path path to the DB files: id_* & pqcode_* & posting_lists_size
 * @param db_prefix the prefix of DB files
 */
template <typename vector_dimension_t> class IndexIVFPQ: public IvfBase <std::vector<uint8_t>, vector_dimension_t>
{
private: /// variables
    // index info
    size_t N_;
    size_t D_;
    size_t L_;
    // level1-index
    size_t kc_;
    size_t mc_;
    size_t dc_;
    // level2-index
    size_t kp_;
    size_t mp_;
    size_t dp_;

    /*==============================================================//
    // Inherit these variables from father.                         //
    // - string name_;                                              //
    // - IndexStatus status_;                                       //
    // - string index_path_;                                        //
    // - string db_path_;                                           //
    // - size_t num_threads                                         //
    // - vector<SegmentClass> segments_;                            //
    // - vector<std::vector<vector_dimension_t>> centers_;          //
    //==============================================================*/

    index::Quantizer<vector_dimension_t> cq_;
    index::Quantizer<vector_dimension_t> pq_;
    std::vector<std::vector<vector_id_t>> posting_lists_; // id for each vectors in segments

    size_t nsamples_; // training sample number
    int seed_;        // seed

    std::vector<std::vector<vector_dimension_t>> centers_cq_;
    // std::vector<cluster_id_t> labels_cq_;

    std::vector<std::vector<std::vector<vector_dimension_t>>> centers_pq_;
    // std::vector<std::vector<uint8_t>> labels_pq_;

    /// const variables
    const std::string cq_centers         = "cq_centers";
    const std::string pq_centers         = "pq_centers";
    const std::string vector_prefix      = "pqcode_";
    const std::string id_prefix          = "id_";


public: /// methods
    explicit IndexIVFPQ(
        size_t N, size_t D, size_t L, 
        size_t kc, 
        size_t kp, size_t mp,  // index config
        const std::string & index_path, 
        const std::string & db_path,
        const std::string & name = "",
        IndexStatus status = IndexStatus::LOCAL
    );

    void Populate(const std::vector<vector_dimension_t> & raw_data) override;

    void Train(const std::vector<vector_dimension_t> & raw_data) override;

    void LoadIndex() override;

    void WriteIndex() override;

    void LoadSegments() override;

    void LoadSegments(const std::vector<cluster_id_t> & book) override;

    void WriteSegments() override;

    void WritePostingLists() const;

    void SetTrainingConfig(size_t nsamples, int seed);

    /// @brief Single Query
    void TopKID (
        size_t k, 
        const std::vector<vector_dimension_t> & query, 
        const std::vector<cluster_id_t> & book,
        std::vector<vector_id_t> & vid,
        std::vector<float> & dist
    ) override;

    /// @brief Batch Queries
    void TopKID (
        size_t k, 
        const std::vector<std::vector<vector_dimension_t>> & queries,
        const std::vector<std::vector<cluster_id_t>> & books,
        std::vector<std::vector<vector_id_t>> & vids,
        std::vector<std::vector<float>> & dists
    ) override;

    /// @brief Single Topw query
    void TopWID (
        size_t w, 
        const std::vector<vector_dimension_t> & query,
        std::vector<cluster_id_t> & book
    ) override;

    /// @brief Batch Topw Queries
    void TopWID (
        size_t w, 
        const std::vector<std::vector<vector_dimension_t>> & queries,
        std::vector<std::vector<cluster_id_t>> & books
    ) override;


private: /// methods
    void InsertIvf(const std::vector<vector_dimension_t> & raw_data);

    /// @brief function for asym-distance, given dtable and pqcode (uint8_t *). 
    inline float ADist(const DistanceTable & dtable, const uint8_t * code) const;

    inline DistanceTable DTable(const std::vector<vector_dimension_t> & vec) const;

    bool Ready(); // ready to topk (all elements ready)

};


/*************************************** 
 * @class Template class declarations. * 
 ***************************************/
template class IndexIVFPQ<uint8_t>;
template class IndexIVFPQ<float>;


};

};

#endif