#include <resize.hpp>

#include <iostream>

using namespace std;

int main()
{
    std::vector<float> v1d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto v2d = utils::Resize<float>::Nest(v1d, 2, 5);

    cout << "v2d: " << endl;
    for (size_t d0 = 0; d0 < v2d.size(); d0++)
    {
        for (size_t d1 = 0; d1 < v2d[d0].size(); d1++)
        {
            cout << v2d[d0][d1] << " ";
        }
        cout << endl;
        v2d[d0].emplace_back(3.14159265359);
    }

    auto v1d_second = utils::Resize<float>::Flatten(v2d);
    cout << "v1d_second: " << endl;
    for (const auto & x: v1d_second)
        cout << x << " ";
    cout << endl;

    auto v3d = utils::Resize<float>::Nest(v1d_second, 3, 2, 2);
    cout << "v3d: " << endl;;
    for (size_t d0 = 0; d0 < v3d.size(); d0++)
    {
        for (size_t d1 = 0; d1 < v3d[d0].size(); d1++)
        {
            for (size_t d2 = 0; d2 < v3d[d0][d1].size(); d2++)
            {
                cout << v3d[d0][d1][d2] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}