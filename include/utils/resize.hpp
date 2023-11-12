#ifndef INCLUDE_RESIZE_HPP
#define INCLUDE_RESIZE_HPP


#include <vector>
#include <stdint.h>


namespace utils {

template <typename vector_dimension_t> class Resize {
public:

    static std::vector<vector_dimension_t> Flatten(const std::vector<std::vector<vector_dimension_t>> & nested);
    
    static std::vector<vector_dimension_t> Flatten(const std::vector<std::vector<std::vector<vector_dimension_t>>> & nested);

    static std::vector<std::vector<vector_dimension_t>> Nest(const std::vector<vector_dimension_t> & flttened, size_t d0, size_t d1);

    static std::vector<std::vector<std::vector<vector_dimension_t>>> Nest(const std::vector<vector_dimension_t> & flattened, size_t d0, size_t d1, size_t d2);

};


template class Resize<uint8_t>;
template class Resize<uint16_t>;
template class Resize<uint32_t>;
template class Resize<uint64_t>;
template class Resize<float>;


};

#endif