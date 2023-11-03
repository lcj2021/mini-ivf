#include <index_ivfpq.hpp>

namespace index {

namespace ivf {


DistanceTable::DistanceTable(size_t M, size_t Ks): kp_(Ks), data_(M * Ks) {}


inline void DistanceTable::set_value(size_t m, size_t ks, float val)
{
    data_[m * kp_ + ks] = val;
}


inline distance_t DistanceTable::get_value(size_t m, size_t ks) const
{
    return data_[m * kp_ + ks];
}


template <typename vector_dimension_t> 
IndexIVFPQ<vector_dimension_t>::IndexIVFPQ(
    size_t N, size_t D, size_t L, size_t kc, size_t mc, size_t dc, // index config
    const std::string & index_path, 
    const std::string & db_path,
    const std::string & name = "",
    IndexStatus status = IndexStatus::LOCAL
): 
N_(N), D_(D), L_(L), kc_(kc), mc_(mc), dc_(dc), 
index_path_(index_path), db_path_(db_path), name_(name), status_(status) 
{}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::Populate(const std::vector<vector_dimension_t> & raw_data)
{
    
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::Train(const std::vector<vector_dimension_t> & raw_data)
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadIndex()
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WriteIndex()
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadSegments()
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadSegments(const std::vector<cluster_id_t> & book)
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WriteSegments()
{

}


template <typename vector_dimension_t> void 
IndexIVFPQ<vector_dimension_t>::InsertIvf(const std::vector<vector_dimension_t> & raw_data)
{

}


template <typename vector_dimension_t> const std::vector<vector_dimension_t> 
IndexIVFPQ<vector_dimension_t>::GetSingleCode(size_t list_no, size_t offset) const
{
    
}

};

};