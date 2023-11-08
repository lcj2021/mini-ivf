#ifndef INCLUDE_INDEX_IVFPQ_HPP
#define INCLUDE_INDEX_IVFPQ_HPP

#include <ivf_base.hpp>
#include <vector>

namespace index {

namespace ivf {


class DistanceTable {
public:
    // Helper structure. This is identical to vec<vec<distance_t>> dt(M, vec<distance_t>(Ks))
    explicit DistanceTable(size_t M, size_t Ks);

    inline void set_value(size_t m, size_t ks, float val);

    inline distance_t get_value(size_t m, size_t ks) const ;

private:
    size_t kp_;
    std::vector<distance_t> data_;
};



/**
 * Configuration structure
 * @param N_ the number of data
 * @param D_ the number of dimensions
 * @param L_ the expected number of candidates involed when searching is performed
 * @param kc the number of coarse quantizer's (nlist) centers. Default: 100
 * @param mc the number of subspace for coarse quantizer. mc must be 1
 * @param dc the dimensions of subspace for coarse quantize and product quantizer. dc must be D_.  dp = D_ / mp
 */
template <typename vector_dimension_t> 

class IndexIVFPQ: public IvfBase <std::vector<vector_dimension_t>, vector_dimension_t>
{
private: /// variables
    size_t N_;
    size_t D_;
    size_t L_;
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
    // - vector<std::vector<vector_dimension_t>> centers_;          //
    //==============================================================*/

    /// @todo quantizer
    
    std::vector<std::vector<vector_id_t>> postint_lists_;

public: /// methods
    IndexIVFPQ(
        size_t N, size_t D, size_t L, size_t kc, size_t mc, size_t dc, // index config
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

private: /// methods
    void InsertIvf(const std::vector<vector_dimension_t> & raw_data);

    const std::vector<vector_dimension_t> GetSingleCode(size_t list_no, size_t offset) const;

    const std::vector<vector_dimension_t> NthRawVector(const std::vector<vector_dimension_t> & long_code, size_t n) const;

};


/**
 * @class Template class declarations.
*/
template class IndexIVFPQ<uint8_t>;
template class IndexIVFPQ<float>;


};

};

#endif