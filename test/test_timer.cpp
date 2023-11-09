#include <timer.hpp>
#include <iostream>
#include <unistd.h>

using namespace std;


int main()
{
    utils::Timer timer;

    timer.Start();
    sleep(3);
    timer.Stop();

    cout << timer.GetTime() << endl;

    timer.Reset();

    timer.Start();
    sleep(1);
    timer.Stop();

    cout << timer.GetTime() << endl;  

    return 0;
}