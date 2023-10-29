#include "util.hpp"

void Timer::Reset() { total = 0; }

void Timer::Start() { t1 = std::chrono::system_clock::now(); }

void Timer::Stop()
{
    t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    total += diff.count();
}

double Timer::GetTime() { return total; }

void ModifyPath(std::string& path) {
    if (path.back() != '/') path += '/';
}

std::string ToStringWithUnits(int nt) {
   if (nt >= 1'000'000'000) {
       return std::to_string(nt / 1'000'000'000) + "b" + ToStringWithUnits(nt % 1'000'000'000);
   } 
   if(nt >= 1'000'000) {
       return std::to_string(nt / 1'000'000) + "m" + ToStringWithUnits(nt % 1'000'000);
   }
   if (nt >= 1'000) {
       return std::to_string(nt / 1'000) + "k";
   }
   return "";
}