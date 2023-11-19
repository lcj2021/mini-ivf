#ifndef INCLUDE_INDEX_IMI_HPP
#define INCLUDE_INDEX_IMI_HPP

#include <ivf/ivf_base.hpp>
#include <ivf/distance_table.hpp>
#include <quantizer.hpp>

#include <vector>


namespace index
{
namespace ivf
{


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
template <typename vector_dimension_t> class IndexIMI: public IvfBase <std::vector<vector_dimension_t>, vector_dimension_t>
{

private: /// variables
    // index info
    size_t N_;
    size_t D_;
    size_t L_;
    // level-1-index
    size_t kc_;
    size_t mc_;
    size_t dc_;
    // level-2-index
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
    //==============================================================*/

    Quantizer<vector_dimension_t> cq_;
    Quantizer<vector_dimension_t> pq_;

    std::vector<std::vector<vector_dimension_t>> centers_cq_;
    std::vector<std::vector<std::vector<vector_dimension_t>>> centers_pq_;

    std::vector<std::vector<vector_id_t>> posting_lists_;   // id for each vectors in segments
    std::vector<std::vector<uint8_t>> pqcodes_;             // PQ code for each segments (used in `Populate`)
    std::vector<std::vector<size_t>> bucket_begin_;         // start point

    size_t nsamples_; // training sample number
    int seed_;        // seed

    size_t num_buckets_;
    
    /// const variables
    const std::string cq_centers_                   = "cq_centers";
    const std::string pq_centers_                   = "pq_centers";
    const std::string bucket_begin_file_            = "bucket_begin_file";
    const std::string posting_lists_size_file_      = "posting_lists_size";
    const std::string vector_prefix_                = "pqcode_";
    const std::string id_prefix_                    = "id_";


public: /// methods
    explicit IndexIMI(
        size_t N, size_t D, size_t L,
        size_t kc,
        size_t kp, size_t mp,
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

    void SetTrainingConfig(size_t nsamples, int seed);

    /// @brief Single Query (1 thread)
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

    /// @brief Single Topw query (1 thread)
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

    /// @brief search for one
    void Search (
        size_t k, size_t w,
        const std::vector<vector_dimension_t> & query,
        std::vector<vector_id_t> & vid,
        std::vector<float> & dist
    ) override;

    /// @brief search for batch
    void Search (
        size_t k, size_t w,
        const std::vector<std::vector<vector_dimension_t>> & queries,
        std::vector<std::vector<vector_id_t>> & vids,
        std::vector<std::vector<float>> & dists
    ) override;


private: /// methods
    void InsertIvf(const std::vector<vector_dimension_t> & raw_data);

    /// @brief function for asym-distance, given dtable and pqcode (uint8_t *). 
    inline float ADist(const DistanceTable & dtable, const uint8_t * code) const;

    inline DistanceTable DTable(const std::vector<vector_dimension_t> & vec) const;

    bool Ready(); // ready to topk (all elements ready)

    inline size_t PQCode2BucketID(const uint8_t * code);

    inline void BucketID2PQCode(size_t bid, std::vector<uint8_t> & code);

};


// Template Class Declaration
template class IndexIMI<uint8_t>;
template class IndexIMI<float>;


}; // namespace ivf

}; // namespace index



#endif