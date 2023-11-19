#ifndef INCLUDE_INDEX_IVFFLAT_HPP
#define INCLUDE_INDEX_IVFFLAT_HPP


#include <ivf/ivf_base.hpp>
#include <quantizer.hpp>
#include <vector>


namespace index {

namespace ivf {


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
template <typename vector_dimension_t> class IndexIVFFLAT: public IvfBase <std::vector<vector_dimension_t>, vector_dimension_t>
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

    /*==============================================================//
    // Inherit these variables from father.                         //
    // - string name_;                                              //
    // - IndexStatus status_;                                       //
    // - string index_path_;                                        //
    // - string db_path_;                                           //
    // - size_t num_threads                                         //
    // - vector<SegmentClass> segments_;                            //
    //==============================================================*/

    std::vector<std::vector<vector_id_t>> posting_lists_;

    size_t nsamples_;
    int seed_;

    Quantizer<vector_dimension_t> cq_;

    std::vector<std::vector<vector_dimension_t>> centers_cq_;

    /// const variables
    const std::string posting_lists_size_file_      = "posting_lists_size"; 
    const std::string cq_centers_                   = "cq_centers";
    const std::string vector_prefix_                = "pqcode_";
    const std::string id_prefix_                    = "id_";

public: /// methods
    explicit IndexIVFFLAT(
        size_t N, size_t D, size_t L,
        size_t kc, // level-1-config
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

private:
    void InsertIvf(const std::vector<vector_dimension_t> & raw_data);

    bool Ready(); // ready to topk (all elements ready)

};


/// @brief Template Class
template class IndexIVFFLAT<uint8_t>;
template class IndexIVFFLAT<float>;


};

};


#endif