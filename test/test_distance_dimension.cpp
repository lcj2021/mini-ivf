#include <distance.hpp>
#include <iostream>

#include <vector>
#include <chrono>

#include <cassert>
#include <cmath>


using namespace std;

using dim_t = uint8_t;

class Timer {
public:
    Timer() : start_timepoint(std::chrono::high_resolution_clock::now()) {}
    
    ~Timer() {
        auto end_timepoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_timepoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_timepoint).time_since_epoch().count();
        auto duration = end - start;
        double milliseconds = duration * 0.001;

        std::cout << milliseconds << endl;
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
    const size_t N = 1000000;
    const size_t D = 64;
    const float err = 1e-2;

    vector<vector<dim_t>> as(N, vector<dim_t> (D));
    vector<vector<dim_t>> bs(N, vector<dim_t> (D));

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < D; j++)
        {
            as[i][j] = rand() % 256;
            bs[i][j] = rand() % 256;
        }
    }

    float gt[N];
    float nos[N];

    for (size_t d = 1; d <= D; d++) {
        { Timer t;
            cout << "brute force ";
            for (size_t i = 0; i < N; i++)
            {
                gt[i] = vec_L2sqr_groudthruth(as[i].data(), bs[i].data(), d);
            }
        }

        { Timer t;
            cout << g_simd_architecture << " [classify] ";
            for (size_t i = 0; i < N; i++)
            {
                nos[i] = vec_L2sqr(as[i].data(), bs[i].data(), d);
            }
        }

        for (size_t i = 0; i < N; i++) {
            assert( fabs(gt[i] - nos[i]) < err );
        }
    }

    return 0;
}