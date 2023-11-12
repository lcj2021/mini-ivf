#ifndef INCLUDE_STIMER_HPP
#define INCLUDE_STIMER_HPP

#include <chrono>

namespace utils {

class STimer
{
private:
    std::chrono::system_clock::time_point t1_;
    double total_;

public:
    STimer();
    void Reset();
    void Start();
    void Stop();
    double GetTime();

};

};

#endif