#ifndef INCLUDE_VECTOR_IO_HPP
#define INCLUDE_VECTOR_IO_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>


namespace utils {


template <typename T> class VectorIO {
public:

    static std::pair<size_t, size_t> LoadFromFile(std::vector<T>& data, const std::string& filename);

    static std::pair<size_t, size_t> LoadFromFile(std::vector<T>& data, const std::string& filename, size_t expect_read_n);

    static void WriteToFile(const std::vector<T>& data, std::pair<size_t, size_t> dimension, const std::string& filename);

};


template class VectorIO<uint8_t>;
template class VectorIO<uint16_t>;
template class VectorIO<uint32_t>;
template class VectorIO<uint64_t>;
template class VectorIO<float>;


};


#endif