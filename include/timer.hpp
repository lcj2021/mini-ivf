#ifndef INCLUDE_TIMER_HPP
#define INCLUDE_TIMER_HPP

#include <chrono>

namespace utils {

class Timer
{
private:
    std::chrono::system_clock::time_point t1_;
    double total_;

public:
    Timer();
    void Reset();
    void Start();
    void Stop();
    double GetTime();

};

};

#endif