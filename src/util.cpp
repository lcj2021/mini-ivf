#include "util.hpp"

void Timer::reset() { total = 0; }

void Timer::start() { t1 = std::chrono::system_clock::now(); }

void Timer::stop()
{
    t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    total += diff.count();
}

double Timer::get_time() { return total; }

void modify_path(std::string& path) {
    if (path.back() != '/') path += '/';
}

std::string to_string_with_units(int nt) {
   if (nt >= 1'000'000'000) {
       return std::to_string(nt / 1'000'000'000) + "b" + to_string_with_units(nt % 1'000'000'000);
   } 
   if(nt >= 1'000'000) {
       return std::to_string(nt / 1'000'000) + "m" + to_string_with_units(nt % 1'000'000);
   }
   if (nt >= 1'000) {
       return std::to_string(nt / 1'000) + "k";
   }
   return "";
}