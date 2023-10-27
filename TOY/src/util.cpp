#include "util.hpp"


Timer::Timer() : total(0) {}

void Timer::reset() { total = 0; }

void Timer::start() { t1 = std::chrono::system_clock::now(); }

void Timer::stop()
{
    t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    total += diff.count();
}

double Timer::get_time() { return total; }
