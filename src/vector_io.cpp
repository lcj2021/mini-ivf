#include <vector_io.hpp>


namespace utils {

// return the dimention of corresponding dataset
template <typename T> std::pair<size_t, size_t> 
VectorIO<T>::LoadFromFile(std::vector<T>& data, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D;
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    size_t N = (file_size) / ((D) * sizeof(T) + sizeof(int));

    file.seekg(0, std::ios::beg); 
    data.resize(N * D);
    data.shrink_to_fit();

    int sep;
    for (size_t n = 0; n < N; ++n) {
        file.read(reinterpret_cast<char*>(&sep), sizeof(int));
        file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(T));
    }
    // printf("%s: [%zu x %d] has loaded!\n", filename.data(), N, D);
    file.close();
    return {N, D};
}



template<typename T> std::pair<size_t, size_t> 
VectorIO<T>::LoadFromFile(std::vector<T>& data, const std::string& filename, size_t expect_read_n) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D;
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    size_t N = (file_size) / ((D) * sizeof(T) + 4);

    assert(expect_read_n <= N);

    file.seekg(0, std::ios::beg); 
    data.resize(expect_read_n * D);
    data.shrink_to_fit();

    int sep;
    for (size_t n = 0; n < expect_read_n; ++n) {
        file.read(reinterpret_cast<char*>(&sep), sizeof(int));
        file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(T));
    }
    // printf("All %s: [%zu x %d] has loaded!\n", filename.data(), expect_read_n, D);
    file.close();
    return {expect_read_n, D};
}



template<typename T> void 
VectorIO<T>::WriteToFile(const std::vector<T>& data, std::pair<size_t, size_t> dimension, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }
    auto [N, D] = dimension;
    assert(data.size() == N * D);

    // char sep[4] = {(char)D, 0, 0, 0};
    int sep = D;
    // 4 + d * sizeof(T) for each vector
    for (size_t n = 0; n < N; ++n) {
        file.write(reinterpret_cast<char*>(&sep), sizeof(int));
        file.write(reinterpret_cast<char*>(const_cast<T*>(data.data()) + n * D), D * sizeof(T));
    }
    // printf("%s: [%zu x %zu] has written!\n", filename.data(), N, D);
    file.close();
}


};