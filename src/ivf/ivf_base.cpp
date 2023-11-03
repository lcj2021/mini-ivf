#include <ivf_base.hpp>

namespace index {

namespace ivf {


template<class SegmentClass, typename vector_dimension_t>
IvfBase<SegmentClass, vector_dimension_t>::IvfBase(
    const std::string & index_path,
    const std::string & db_path,
    const std::string & name,
    IndexStatus status
): index_path_(index_path), db_path_(db_path), name_(name), status_(status) 
{
    num_threads_ = 1; // By default.
}


template<class SegmentClass, typename vector_dimension_t> size_t
IvfBase<SegmentClass, vector_dimension_t>::GetNumThreads() const
{
    return num_threads_;
}


template<class SegmentClass, typename vector_dimension_t> void 
IvfBase<SegmentClass, vector_dimension_t>::SetNumThreads(size_t num_threads)
{
    num_threads_ = num_threads;
}


template<class SegmentClass, typename vector_dimension_t> std::string
IvfBase<SegmentClass, vector_dimension_t>::GetName() const 
{ 
    return name_; 
}


template<class SegmentClass, typename vector_dimension_t> std::string
IvfBase<SegmentClass, vector_dimension_t>::GetIndexPath() const
{
    return index_path_;
}


template<class SegmentClass, typename vector_dimension_t> std::string
IvfBase<SegmentClass, vector_dimension_t>::GetDBPath() const
{
    return db_path_;
}


template<class SegmentClass, typename vector_dimension_t> void
IvfBase<SegmentClass, vector_dimension_t>::SetName(const std::string & name)
{
    name_ = name;
}


template<class SegmentClass, typename vector_dimension_t> void
IvfBase<SegmentClass, vector_dimension_t>::SetIndexPath(const std::string & index_path)
{
    index_path_ = index_path;
}


template<class SegmentClass, typename vector_dimension_t> void
IvfBase<SegmentClass, vector_dimension_t>::SetDBPath(const std::string & db_path)
{
    db_path_ = db_path;
}

};

};
