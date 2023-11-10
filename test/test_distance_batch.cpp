#include <distance.hpp>
#include <iostream>

#include <vector>
#include <chrono>


using namespace std;

using dim_t = float;

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


float vec_L2sqr_groudthruth(const dim_t* a, const dim_t* b, size_t dim)
{
    float dist = 0;
    for (size_t i = 0; i < dim; i++) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return dist;
}


int main()
{
    const size_t N = 100000;
    const size_t d = 2;

    vector<vector<dim_t>> as(N, vector<dim_t> (d));
    vector<vector<dim_t>> bs(N, vector<dim_t> (d));

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < d; j++)
        {
            as[i][j] = rand() % 256;
            bs[i][j] = rand() % 256;
        }
    }

    float gt[N + num_float_per_simd_vector];
    float bds[N + num_float_per_simd_vector];

    { Timer t;
        cout << "brute force" << endl;
        for (size_t i = 0; i < N; i++)
        {
            gt[i] = vec_L2sqr_groudthruth(as[i].data(), bs[i].data(), d);
        }
    }

    { Timer t;
        cout << g_simd_architecture << endl;
        for (size_t i = 0; i < N - 1; i++)
        {
            vec_L2sqr_batch(as[i].data(), bs[i].data(), d, bds + i);
        }
        vec_L2sqr_batch(as.back().data(), bs.back().data(), d, bds + N - 1);
    }

    return 0;
}