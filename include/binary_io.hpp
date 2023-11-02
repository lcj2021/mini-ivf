#ifndef BINARY_IO_H
#define BINARY_IO_H

#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

/**
 * LoadFromFileBinary<type in>(type out) 
 * return the dimention of corresponding dataset
 * @param data: vector that load into
 * @param filename: path of binary file
*/
template<typename Tin, typename Tout>
std::pair<size_t, size_t> LoadFromFileBinary(std::vector<Tout>& data, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D;
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    size_t N = (file_size) / ((D) * sizeof(Tin) + 4);

    file.seekg(0, std::ios::beg); 
    data.resize(N * D);
    data.shrink_to_fit();

    float sep;
    if (std::is_same<Tin, Tout>()) {
        for (size_t n = 0; n < N; ++n) {
            file.read(reinterpret_cast<char*>(&sep), 4);
            file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(Tout));
        }
    } else {
        std::vector<Tin> buffer(D);
        for (size_t n = 0; n < N; ++n) {
            file.read(reinterpret_cast<char*>(&sep), 4);
            file.read(reinterpret_cast<char*>(buffer.data()), D * sizeof(Tin));
            for (size_t d = 0; d < D; ++d) {
                data[n * D + d] = static_cast<Tout>(buffer[d]);
            }
        }
    }
    printf("%s: [%zu x %d] has loaded!\n", filename.data(), N, D);
    file.close();
    return {N, D};
}

template<typename Tin, typename Tout>
std::pair<size_t, size_t> LoadFromFileBinary(std::vector<Tout>& data, const std::string& filename, size_t expect_read_n) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D;
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    size_t N = (file_size) / ((D) * sizeof(Tin) + 4);

    file.seekg(0, std::ios::beg); 
    data.resize(expect_read_n * D);
    data.shrink_to_fit();

    assert(expect_read_n <= N);

    float sep;
    if (std::is_same<Tin, Tout>()) {
        for (size_t n = 0; n < expect_read_n; ++n) {
            file.read(reinterpret_cast<char*>(&sep), 4);
            file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(Tout));
            if (n % 1000'000 == 0) {
                printf("%zu vectors has loaded!\n", n);
            }
        }
    } else {
        std::vector<Tin> buffer(D);
        for (size_t n = 0; n < expect_read_n; ++n) {
            file.read(reinterpret_cast<char*>(&sep), 4);
            file.read(reinterpret_cast<char*>(buffer.data()), D * sizeof(Tin));
            for (size_t d = 0; d < D; ++d) {
                data[n * D + d] = static_cast<Tout>(buffer[d]);
            }
        }
    }
    printf("All %s: [%zu x %d] has loaded!\n", filename.data(), expect_read_n, D);
    file.close();
    return {expect_read_n, D};
}

template<typename T>
std::pair<size_t, size_t> LoadFromFileBinaryDeep(std::vector<T>& data, const std::string& filename, size_t expect_read_n) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D, N;
    file.read(reinterpret_cast<char*>(&N), sizeof(int));
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    // std::cout << D << ' ' << N << ' ' << file_size << '\n';
    // assert((file_size - 8) / (D * sizeof(float)) == N);

    assert(expect_read_n <= N);

    file.seekg(0, std::ios::beg); 
    data.resize(expect_read_n * D);
    data.shrink_to_fit();

    float sep;
    for (size_t n = 0; n < expect_read_n; ++n) {
        // file.read(reinterpret_cast<char*>(&sep), 4);
        file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(float));
        if (n % 1000'000 == 0) {
            printf("%zu vectors has loaded!\n", n);
        }
    }
    printf("All %s: [%zu x %d] has loaded!\n", filename.data(), expect_read_n, D);
    file.close();
    return {N, D};
}

template<typename T>
void WriteToFileBinary(const std::vector<T>& data, std::pair<size_t, size_t> dimension, const std::string& filename) {
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
        file.write(reinterpret_cast<char*>(&sep), 4);
        file.write(reinterpret_cast<char*>(const_cast<T*>(data.data()) + n * D), D * sizeof(T));
    }
    printf("%s: [%zu x %zu] has written!\n", filename.data(), N, D);
    file.close();
}

#endif