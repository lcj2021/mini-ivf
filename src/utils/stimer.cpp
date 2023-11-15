#include <utils/stimer.hpp>


namespace utils {


/// @brief A tiny timer to test runtime in "second" unit.
STimer::STimer() : total_(0) {}

void STimer::Reset() { total_ = 0; }

void STimer::Start() { t1_ = std::chrono::system_clock::now(); }

void STimer::Stop()
{
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = t2 - t1_;
    total_ += diff.count();
}

double STimer::GetTime() { return total_; }


};