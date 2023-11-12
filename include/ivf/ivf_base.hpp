#ifndef INCLUDE_IVF_BASE_HPP
#define INCLUDE_IVF_BASE_HPP

#include <vector>
#include <string>
#include <index.hpp>

namespace index {

namespace ivf {


template <class SegmentClass, typename vector_dimension_t>
class IvfBase {

/// @attention Basic protected variables
protected: 
    std::string name_;
    IndexStatus status_;
    std::string index_path_;
    std::string db_path_;

    std::vector<SegmentClass> segments_;
    
    size_t num_threads_;

public:
    IvfBase(
        const std::string & index_path,
        const std::string & db_path,
        const std::string & name,
        IndexStatus status
    );

    /// @attention Basic public virtual methods
    virtual void Populate(const std::vector<vector_dimension_t> & raw_data) = 0;

    virtual void Train(const std::vector<vector_dimension_t> & raw_data) = 0;

    virtual void LoadIndex() = 0;

    /// @brief Write index file (Assert status == IndexStatus::LOCAL)
    virtual void WriteIndex() = 0;

    /// @brief  Load all segments
    virtual void LoadSegments() = 0;

    /// @brief Load partial segments (status for either LOCAL and DISTRIBUTED)
    virtual void LoadSegments(const std::vector<cluster_id_t> & book) = 0;

    /// @brief Write all segments (Assert status == IndexStatus::LOCAL)
    virtual void WriteSegments() = 0;

    size_t GetNumThreads() const;

    std::string GetName() const;

    std::string GetIndexPath() const;

    std::string GetDBPath() const;

    void SetNumThreads(size_t num_threads);

    void SetName(const std::string & name);

    void SetIndexPath(const std::string & index_path);

    void SetDBPath(const std::string & db_path);

    /// @brief Single Query
    virtual void TopKID (
        size_t k, 
        const std::vector<vector_dimension_t> & query, 
        const std::vector<cluster_id_t> & book,
        std::vector<vector_id_t> & vid,
        std::vector<float> & dist
    ) = 0;

    /// @brief Batch Queries
    virtual void TopKID (
        size_t k, 
        const std::vector<std::vector<vector_dimension_t>> & queries,
        const std::vector<std::vector<cluster_id_t>> & books,
        std::vector<std::vector<vector_id_t>> & vids,
        std::vector<std::vector<float>> & dists
    ) = 0;

    /// @brief Single Topw query
    virtual void TopWID (
        size_t w, 
        const std::vector<vector_dimension_t> & query,
        std::vector<cluster_id_t> & book
    ) = 0;

    /// @brief Batch Topw Queries
    virtual void TopWID (
        size_t w, 
        const std::vector<std::vector<vector_dimension_t>> & queries,
        std::vector<std::vector<cluster_id_t>> & books
    ) = 0;

};



};

};

#endif