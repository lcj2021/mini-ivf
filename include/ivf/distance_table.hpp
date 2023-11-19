#ifndef INCLUDE_DISTANCE_TABLE_HPP
#define INCLUDE_DISTANCE_TABLE_HPP

#include <vector>


namespace index
{
    
namespace ivf
{
    
// Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
class DistanceTable {
public:
    // Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
    explicit DistanceTable(size_t M, size_t Ks);

    void SetValue(size_t m, size_t ks, float val);

    float GetValue(size_t m, size_t ks) const ;

private:
    size_t kp_;
    std::vector<float> data_;
    
};

}; // namespace ivf


}; // namespace index



#endif