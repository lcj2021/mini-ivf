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
    std::vector<std::vector<vector_dimension_t>> centers_;
    
    size_t num_threads_;

public:
    IvfBase(
        const std::string & index_path,
        const std::string & db_path,
        const std::string & name,
        IndexStatus status
    );

    /// @attention Basic public virtual methods
    void Populate(const std::vector<vector_dimension_t> & raw_data) = 0;

    void Train(const std::vector<vector_dimension_t> & raw_data) = 0;

    void LoadIndex() = 0;

    /// @brief Write index file (Assert status == IndexStatus::LOCAL)
    void WriteIndex() = 0;

    /// @brief  Load all segments (Assert status == IndexStatus::LOCAL)
    void LoadSegments() = 0;

    /// @brief Load partial segments (status for either LOCAL and DISTRIBUTED)
    void LoadSegments(const std::vector<cluster_id_t> & book) = 0;

    /// @brief Write all segments (Assert status == IndexStatus::LOCAL)
    void WriteSegments() = 0;

    size_t GetNumThreads() const;

    std::string GetName() const;

    std::string GetIndexPath() const;

    std::string GetDBPath() const;

    void SetNumThreads(size_t num_threads);

    void SetName(const std::string & name);

    void SetIndexPath(const std::string & index_path);

    void SetDBPath(const std::string & db_path);

    /// @brief Single Query
    std::vector<vector_id_t> TopKID(size_t k, const std::vector<vector_dimension_t> & query) = 0;

    /// @brief Batch Queries
    std::vector<std::vector<vector_id_t>> TopKID(size_t k, const std::vector<std::vector<vector_dimension_t>> & queries) = 0;

    /// @brief Single Topw query
    std::vector<cluster_id_t> TopWID(size_t w, const std::vector<vector_dimension_t> & query) = 0;

    /// @brief Batch Topw Queries
    std::vector<std::vector<cluster_id_t>> TopWID(size_t w, const std::vector<std::vector<vector_dimension_t>> & queries) = 0;

};



};

};

#endif