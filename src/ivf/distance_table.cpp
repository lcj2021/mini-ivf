#include <ivf/distance_table.hpp>


namespace index
{

namespace ivf
{

DistanceTable::DistanceTable(size_t M, size_t Ks): kp_(Ks), data_(M * Ks) {}


void DistanceTable::SetValue(size_t m, size_t ks, float val)
{
    data_[m * kp_ + ks] = val;
}


float DistanceTable::GetValue(size_t m, size_t ks) const
{
    return data_[m * kp_ + ks];
}
  
}; // namespace ivf


}; // namespace index

