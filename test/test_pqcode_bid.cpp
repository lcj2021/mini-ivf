#include <vector>
#include <iostream>

using namespace std;

int main()
{
    const size_t mp = 3, kp = 2;
    size_t nb = 1;
    for (size_t i = 0; i < mp; i++)
    {
        nb *= kp;
    }
    vector<vector<int>> v(mp, vector<int>(kp));

    for (size_t i = 0; i < mp; ++i) {
        for (size_t j = 0; j < kp; ++j) {
            v[i][j] = rand() % 10;
            cout << v[i][j] << " ";
        }
        cout << endl;
    }

    cout << "=======================================" << endl;

    vector<int> s(nb);
    {
        vector<size_t> code(mp, 0);
        int dist = 0;
        for (size_t i = 0; i < mp; i++)
        {
            dist += v[i][0];
        }

        for (size_t bid = 0; bid < nb; bid++)
        {
            s[bid] = dist;
            dist = dist - v[0][code[0]] + v[0][(++code[0])%kp];
            for (size_t i = 1; i < mp; i++)
            {
                if (code[i - 1] / kp) {
                    code[i - 1] = 0;
                    dist = dist - v[i][code[i]] + v[i][(++code[i])%kp];
                }
            }
        }
    }

    for (const auto & x: s) cout << x << " ";
    cout << endl;

    return 0;
}