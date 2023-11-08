#include <timer.hpp>


namespace utils {


void Timer::Reset() { total_ = 0; }

void Timer::Start() { t1_ = std::chrono::system_clock::now(); }

void Timer::Stop()
{
    std::chrono::duration<double> diff_ = std::chrono::system_clock::now() - t1_;
    total_ += diff.count();
}

double Timer::GetTime() { return total_; }


};