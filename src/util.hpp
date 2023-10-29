#ifndef UTIL_H
#define UTIL_H

#include <chrono>
#include <vector>
#include <cassert>
#include <string>

class Timer
{
  private:
    std::chrono::system_clock::time_point t1, t2;
    double total;

  public:
    Timer() : total(0) {}
    void Reset();
    void Start();
    void Stop();
    double GetTime();
};

template<typename T>
std::vector<T> flatten(std::vector<std::vector<std::vector<T>>> nested)
{
    size_t d0 = nested.size();
    size_t d1 = nested[0].size();
    size_t d2 = nested[0][0].size();

    std::vector<T> flattened(d0 * d1 * d2);
    size_t cursor = 0;
    for (const auto& x0 : nested) {
        for (const auto& x1 : x0) {
            for (const auto& x2 : x1) {
                flattened[cursor ++] = x2;
            }
        }
    }
    return flattened;
}

template<typename T>
std::vector<std::vector<std::vector<T>>> 
nest(const std::vector<T>& flattened, const std::vector<size_t>& shape)
{
    assert(shape.size() == 3);
    auto d0 = shape[0];
    auto d1 = shape[1];
    auto d2 = shape[2];
    assert(flattened.size() == d0 * d1 * d2);

    std::vector<std::vector<std::vector<T>>> nested(d0, 
        std::vector<std::vector<T>>(d1, 
        std::vector<T>(d2)));

    size_t cursor = 0;
    for (size_t i0 = 0; i0 < d0; ++i0) {
        for (size_t i1 = 0; i1 < d1; ++i1) {
            for (size_t i2 = 0; i2 < d2; ++i2) {
                nested[i0][i1][i2] = flattened[cursor ++];
            }
        }
    }
    return nested;
}

template<typename T>
std::vector<std::vector<T>>
nest_2d(const std::vector<T>& flattened, const std::vector<size_t>& shape)
{
    assert(shape.size() == 2);
    auto d0 = shape[0];
    auto d1 = shape[1];
    assert(flattened.size() == d0 * d1);

    std::vector<std::vector<T>> nested(d0, 
        std::vector<T>(d1));

    size_t cursor = 0;
    for (size_t i0 = 0; i0 < d0; ++i0) {
        for (size_t i1 = 0; i1 < d1; ++i1) {
            nested[i0][i1] = flattened[cursor ++];
        }
    }
    return nested;
}

void ModifyPath(std::string& path);
std::string ToStringWithUnits(int nt);

#endif