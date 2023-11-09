#include <timer.hpp>


namespace utils {


/// @brief A tiny timer to test runtime in "second" unit.
Timer::Timer() : total_(0) {}

void Timer::Reset() { total_ = 0; }

void Timer::Start() { t1_ = std::chrono::system_clock::now(); }

void Timer::Stop()
{
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - t1_;
    total_ += diff.count();
}

double Timer::GetTime() { return total_; }


};