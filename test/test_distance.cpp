// #undef __AVX512F__
// #undef __AVX__

#include <distance.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

float vec_L2sqr_groudthruth(const uint8_t* a, const uint8_t* b, size_t dim)
{
    float dist = 0;
    for (size_t i = 0; i < dim; i++) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return dist;
}

float vec_L2sqr_groudthruth(const float* a, const float* b, size_t dim)
{
    float dist = 0;
    for (size_t i = 0; i < dim; i++) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return dist;
}


class Timer {
public:
    Timer() : start_timepoint(std::chrono::high_resolution_clock::now()) {}
    
    ~Timer() {
        auto end_timepoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_timepoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_timepoint).time_since_epoch().count();
        auto duration = end - start;
        double milliseconds = duration * 0.001;

        std::cout << "Time taken: " << milliseconds << " ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_timepoint;
};


int main()
{
    size_t dim = 960;
    vector<float> a(dim), b(dim);

    for (size_t i = 0; i < dim; i++)
    {
        a[i] = rand();
        b[i] = rand();
    }

    { Timer t;
        cout << "brute force" << endl;
        cout << vec_L2sqr_groudthruth(a.data(), b.data(), dim) << endl;
    }

    { Timer t;
        cout << g_simd_architecture << endl;
        cout << vec_L2sqr(a.data(), b.data(), dim) << endl;
    }

    return 0;
}
